# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 17:29:17 2022

@author: admin
"""
import random
from importlib import import_module
import torch
import os
import json
import numpy as np
from option import args
import common
from trainer import Trainer

# 定义主函数 net_main，用于神经网络的训练和测试
# 参数说明：
# data_loader_path: 数据加载器的路径
# model_name: 模型的名称
# batch_size: 训练或测试时的批量大小
# epoch: 训练的轮数
# run_option: 运行选项，可选值为 'train', 'test', 'resume'
# load_model_path: 加载模型的路径
# lr: 学习率
# save_file: 保存文件的路径
def net_main(data_loader_path, model_name, batch_size, epoch, run_option, load_model_path, lr, save_file):
    # 将数据加载器路径赋值给 args 对象的 data_loader_path 属性
    args.data_loader_path = data_loader_path
    # 将模型名称赋值给 args 对象的 model 属性
    args.model = model_name
    # 将批量大小赋值给 args 对象的 batch_size 属性
    args.batch_size = batch_size
    # 将训练轮数赋值给 args 对象的 epoch 属性
    args.epoch = epoch
    # 打开配置文件 config.json 并读取内容
    config = json.loads(open(os.path.join(args.data_loader_path, 'config.json'), 
                                 'r', encoding='utf-8').read())
    # 获取频率点数
    num_frequency = config['Frequency']['NumValues']
    # 根据运行选项设置 args 对象的 resume 和 test_only 属性
    if run_option == 'train':
        # 训练模式下，不恢复训练，不仅进行测试
        args.resume = False
        args.test_only = False
    elif run_option == 'test':
        # 测试模式下，不恢复训练，仅进行测试
        args.resume = False
        args.test_only = True
    elif run_option == 'resume':
        # 恢复训练模式，此处代码未完成，后续可能需要补充相关逻辑
        args.resume = True
        args.test_only = False

    args.load_model_path = load_model_path
    args.lr = lr
    args.save_file = save_file

    # 设置随机种子，保证实验结果的可重复性
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # 创建 CheckPoint 对象，用于记录训练过程中的信息
    checkpoint = common.CheckPoint(args)
    # 动态导入模型模块
    module = import_module('model.' + args.model.lower())
    # 根据频率点数创建当前模型
    current_model = module.make_model(num_frequency)
    # 创建 Trainer 对象，用于模型的训练和测试
    t = Trainer(args, current_model, checkpoint)
    # 如果不是仅测试模式，不是仅实验模式，也不是仅绘图模式
    if (not args.test_only) and (not args.exp_only) and (not args.plot_only):
        # 调用 Trainer 的 train 方法进行模型训练，获取模型保存路径
        model_path = t.train(args.load_model_path)
        # 调用 Trainer 的 test 方法对训练好的模型进行测试
        t.test(model_path)
        # 从模型路径中提取模型保存名称（去除文件扩展名）
        model_save_name = (str.split(model_path, '/')[-1])[:-4]
    # 如果是仅测试模式
    elif args.test_only:
        # 调用 Trainer 的 test 方法对指定路径的模型进行测试
        t.test(args.load_model_path)
        # 从指定的模型加载路径中提取模型保存名称（去除文件扩展名）
        model_save_name = (str.split(args.load_model_path, '/')[-1])[:-4]
    # 关闭日志文件
    checkpoint.done()
    # 返回模型保存名称
    return model_save_name
