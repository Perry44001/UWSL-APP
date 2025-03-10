# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 17:29:17 2022

@author: admin
"""
# 导入随机数生成模块
import random
# 导入模块导入工具
from importlib import import_module

# 导入科学计算库
import numpy as np
# 导入深度学习框架 PyTorch
import torch
# 导入自定义的 common 模块
import common
# 导入 JSON 模块
import json
# 导入操作系统模块
import os
# 从 option 模块中导入 args 对象
from option import args
# 从 trainer 模块中导入 Trainer 类
from trainer import Trainer


if __name__ == '__main__':
    # 再次设置随机种子，确保可复现性
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.load_model_path = args.load_model_path.replace("\\", '/')
    args.save_file = args.save_file.replace("\\", '/')
    args.data_loader_path = args.data_loader_path.replace("\\", '/')
    config = json.loads(open(os.path.join(args.data_loader_path, 'config.json'), 
                                 'r', encoding='utf-8').read())
    # 获取频率点数
    num_frequency = config['Frequency']['NumValues']
    if args.save_file == '-':
        args.save_file = args.data_loader_path
    # 创建检查点对象，用于记录训练过程
    checkpoint = common.CheckPoint(args)
    # 动态导入模型模块
    module = import_module('model.' + args.model.lower())
    # 根据参数创建当前模型
    current_model = module.make_model(num_frequency)
    # 创建训练器对象
    t = Trainer(args, current_model, checkpoint)
    # 如果不是仅测试、仅实验、仅绘图模式
    if (not args.test_only) and (not args.exp_only) and (not args.plot_only):
        # 调用训练器的训练方法进行训练，并返回模型保存路径
        model_path = t.train(args.load_model_path)
        # 调用训练器的测试方法进行测试
        t.test(model_path)
        # 提取模型保存名称
        model_save_name = (str.split(model_path, '/')[-1])[:-4]
    elif args.test_only:
        # 如果是仅测试模式，调用训练器的测试方法进行测试
        t.test(args.load_model_path)
        # 提取模型保存名称
        model_save_name = (str.split(args.load_model_path, '/')[-1])[:-4]

    # 完成检查点操作
    checkpoint.done()