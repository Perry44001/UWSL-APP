# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 16:48:02 2022

@author: admin
"""
import os
import json
# import math
# from decimal import Decimal
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
import time
import pylab as pl
from Plt.beaplot import plot as bp
from data.Dataloader import SnSpectrumLoader
import scipy.io as sio
import glob

class Trainer():
    def __init__(self, args, my_model, ckp):
        """
        初始化Trainer类的实例。
    
        :param args: 包含命令行参数的对象，用于配置训练过程。
        :param my_model: 神经网络模型实例，将在训练过程中使用。
        :param ckp: 检查点对象，用于保存和恢复训练状态。
        """
        # 保存命令行参数
        self.args = args
        # 保存检查点对象
        self.ckp = ckp
        # 保存神经网络模型
        self.model = my_model
        # 从配置文件中加载配置信息
        # 打开位于数据加载路径下的config.json文件
        config = json.loads(open(os.path.join(self.args.data_loader_path, 'config.json'), 
                                 'r', encoding='utf-8').read())
        # 获取接收器的数量
        self.num_receiver = len(config['ReceiverDepth'])
        # 获取频率点数
        self.num_frequency = config['Frequency']['NumValues']
        # 获取训练集中源的最大距离
        self.r_max = config['trainset']['SourceRange']['UpperLimit']
        # 获取训练集中源的最大深度
        self.z_max = config['trainset']['SourceDepth']['UpperLimit']
    def train(self, save_name_weight_para_file, mode='train'):
        """
        训练神经网络模型。
    
        :param save_name_weight_para_file: 保存权重参数文件的名称
        :param mode: 训练模式，默认为 'train'
        :return: 保存权重参数文件的路径
        """
        # 使用GPU加速计算过程
        torch.cuda.empty_cache()
        # 选择设备，如果有可用的CUDA则使用GPU，否则使用CPU
        default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("running at device %s" % default_device)
        # 设置默认数据类型为32位浮点数
        default_type = torch.float32
        # 将模型移动到指定设备并设置数据类型
        self.model = self.model.to(default_device).type(default_type)

        # 训练集、测试集、实验数据和模型文件的路径
        model_name = self.args.model

        # 旧的训练集路径，已注释
        # path = os.path.abspath('../1.1 Dataset Simulation/1.1.3 Dataset/1.1.3.1 Training Set/' + self.args.data_train)
        # 获取训练集路径
        path = os.path.abspath(self.args.data_loader_path + '/A.TrainingSet')
        # 获取验证集路径，这里与训练集路径相同
        path_v = os.path.abspath(self.args.data_loader_path + '/A.TrainingSet')
        # 获取训练集目录下所有以.sim结尾的文件
        file_name_all = glob.glob(path + '/*.sim')
        # 获取数据加载路径的最后一级目录名
        appendix = str.split(self.args.data_loader_path, '/')[-1]

        # 计算所有训练集文件的总大小
        data_size = os.path.getsize(file_name_all[0]) * len(file_name_all)

        # 计算一个样本数据的大小
        one_sample_data_size = 2 * self.num_frequency * self.num_receiver + 2  # One sample data size

        # 获取训练批次大小
        batch_size = self.args.batch_size  # Training batch size
        # 获取验证批次大小，这里与训练批次大小相同
        batch_size_val = self.args.batch_size  # Validation batch size

        # 模型初始化
        # 遍历模型的所有模块
        for m in self.model.modules():
            # 如果是卷积层或全连接层，使用Xavier均匀分布初始化权重
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        # 打印模型的总参数数量
        print("Total number of parameters in networks is {} ".format(
            sum(x.numel() for x in self.model.parameters())))  # Print number of network parameters

        # 设置优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.99), weight_decay=5e-5)

        # 加载模型或重置模型
        if self.args.resume:
            # 断点路径
            path_checkpoint = save_name_weight_para_file
            # 加载断点
            checkpoint = torch.load(path_checkpoint)
            # 加载模型的可学习参数
            self.model.load_state_dict(checkpoint['model'])
            # 加载优化器的参数
            optimizer.load_state_dict(checkpoint['optimizer'])
            # 获取小周期数
            mini_epoch = checkpoint['mini_epoch']
            # 获取最大周期数
            max_epoch = checkpoint['max_epoch']
            # 获取当前周期数，并加1作为起始周期
            start_epoch = checkpoint['current_epoch'] + 1

            # 按周期监控的损失和平均绝对误差
            training_loss = checkpoint['training_loss']
            validation_loss = checkpoint['validation_loss']
            # 训练过程中的变量
            log_sigma_of_range = checkpoint['log_sigma_of_range']
            log_sigma_of_depth = checkpoint['log_sigma_of_depth']
            MAPE_of_range = checkpoint['MAPE_of_range']
            MAPE_of_depth = checkpoint['MAPE_of_depth']
            MAE_of_range = checkpoint['MAE_of_range']
            MAE_of_depth = checkpoint['MAE_of_depth']
            # 验证过程中的变量
            MAPE_of_range_v = checkpoint['MAPE_of_range_v']
            MAPE_of_depth_v = checkpoint['MAPE_of_depth_v']
            MAE_of_range_v = checkpoint['MAE_of_range_v']
            MAE_of_depth_v = checkpoint['MAE_of_depth_v']

        else:
            # 从第0个周期开始
            start_epoch = 0
            # 初始化模型保存路径
            save_name_weight_para_file = ''
            epoch = self.args.epoch
            # 获取小周期数
            if epoch < 1000:
                mini_epoch = epoch
                max_epoch = 1
            else:
                mini_epoch = int(epoch/10)
                max_epoch = 10
            # 按周期监控的损失和平均绝对误差
            training_loss = []
            validation_loss = []
            # 训练过程中的变量
            log_sigma_of_range = []
            log_sigma_of_depth = []
            MAPE_of_range = []
            MAPE_of_depth = []
            MAE_of_range = []
            MAE_of_depth = []
            # 验证过程中的变量
            MAPE_of_range_v = []
            MAPE_of_depth_v = []
            MAE_of_range_v = []
            # 停止误差
            MAE_of_depth_v = []

        # 计算每个小周期的批次索引数量
        batch_ndx = int(data_size / one_sample_data_size / 4 / batch_size / mini_epoch)
        # 获取源的数量
        num_of_sources = self.args.num_of_sources
        # 计算每个源的批次数量
        Nba = int(batch_ndx * batch_size / num_of_sources)

        # 开始训练
        for current_epoch in range(start_epoch, max_epoch * mini_epoch):

            # 设置训练过程中的信噪比
            if current_epoch <= 1:
                # 在第一个周期，希望信噪比小一些
                SNR_range_of_train = [-10, 10]
                SNR_range_of_validation = [-20, 10]
            else:
                SNR_range_of_train = [20, 30]
                SNR_range_of_validation = [10, 20]

            # 初始化参数
            loss = 0
            MAPE_r = 0  # 训练过程中距离估计的平均绝对百分比误差
            MAPE_d = 0  # 训练过程中深度估计的平均绝对百分比误差
            MAE_r = 0  # 训练过程中距离估计的平均绝对误差 (km)
            MAE_d = 0  # 训练过程中深度估计的平均绝对误差 (m)
            loss_v = 0
            MAPE_r_v = 0  # 验证过程中距离估计的平均绝对百分比误差
            MAPE_d_v = 0  # 验证过程中深度估计的平均绝对百分比误差
            MAE_r_v = 0  # 验证过程中距离估计的平均绝对误差 (km)
            MAE_d_v = 0  # 验证过程中深度估计的平均绝对误差 (m)

            for iba in range(0, Nba):

                # 数据加载
                train_set = SnSpectrumLoader(file_path=path, length_freq=self.num_frequency,
                                             SNR_range=SNR_range_of_train,
                                             run_mode=mode, model=model_name, num_of_receiver=self.num_receiver)
                train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=self.args.num_workers,
                                               drop_last=True)
                val_set = SnSpectrumLoader(file_path=path_v, length_freq=self.num_frequency,
                                           SNR_range=SNR_range_of_validation,
                                           run_mode=mode, model=model_name, num_of_receiver=self.num_receiver)
                val_loader = data.DataLoader(val_set, batch_size=batch_size_val, shuffle=True,
                                             num_workers=self.args.num_workers,
                                             drop_last=True)
                for batch_idx_mini, dataTrain in enumerate(train_loader):

                    batch_idx = batch_idx_mini + iba * len(train_loader)
                    # 设置模型为训练模式
                    self.model.train()

                    # 导入数据
                    inputs = dataTrain['C'].float()
                    inputs = Variable(inputs)
                    inputs = inputs.to(default_device).type(default_type)
                    r = dataTrain['r'].float() / self.r_max  # 距离目标
                    r = r.to(default_device).type(default_type)
                    d = dataTrain['z'].float() / self.z_max  # 深度目标
                    d = d.to(default_device).type(default_type)
                    # 清空优化器的梯度缓存
                    optimizer.zero_grad()
                    # 将数据输入网络进行训练
                    try:
                        # 前向传播，计算训练损失、对数方差和输出
                        train_loss, log_vars, output = self.model(inputs, [r, d])
                    except RuntimeError as e:
                        # 处理内存不足的异常
                        if 'out of memory' in str(e):
                            print('|WARNING: run out of memory')
                            # 尝试清空CUDA缓存
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                        else:
                            # 抛出其他异常
                            raise e
                    # 反向传播，计算梯度
                    train_loss.backward()
                    # 更新模型参数
                    optimizer.step()
                    # 计算误差
                    # 计算训练过程中距离估计的平均绝对百分比误差
                    MAPE_r += float(((output[0] - r).abs() / r).sum(0) / batch_size)
                    # 计算训练过程中深度估计的平均绝对百分比误差
                    MAPE_d += float(((output[1] - d).abs() / d).sum(0) / batch_size)
                    # 计算训练过程中距离估计的平均绝对误差 (km)
                    MAE_r += float((output[0] - r).abs().sum(0) / batch_size * self.r_max)
                    # 计算训练过程中深度估计的平均绝对误差 (m)
                    MAE_d += float((output[1] - d).abs().sum(0) / batch_size * self.z_max)
                    # 累加训练损失
                    loss += float(train_loss.item())
                    # 验证过程
                    with torch.no_grad():
                        # 设置模型为评估模式
                        self.model.eval()
                        # 获取验证集的一个批次数据
                        _, dataVali = list(enumerate(val_loader))[batch_idx_mini]
                        # 提取验证集的输入数据
                        inputs_v = dataVali['C'].float()
                        inputs_v = Variable(inputs_v)
                        inputs_v = inputs_v.to(default_device).type(default_type)
                        # 提取验证集的距离目标
                        r_v = dataVali['r'].float() / self.r_max
                        r_v = r_v.to(default_device).type(default_type)
                        # 提取验证集的深度目标
                        z_v = dataVali['z'].float() / self.z_max
                        z_v = z_v.to(default_device).type(default_type)
                        # 计算验证损失、对数方差和输出
                        val_loss, log_vars_v, output_v = self.model(inputs_v, [r_v, z_v])
                        # 计算验证过程中距离估计的平均绝对百分比误差
                        MAPE_r_v += float(((output_v[0] - r_v).abs() / r_v).sum(0) / batch_size_val)
                        # 计算验证过程中深度估计的平均绝对百分比误差
                        MAPE_d_v += float(((output_v[1] - z_v).abs() / z_v).sum(0) / batch_size_val)
                        # 计算验证过程中距离估计的平均绝对误差 (km)
                        MAE_r_v += float((output_v[0] - r_v).abs().sum(0) / batch_size_val * self.r_max)
                        # 计算验证过程中深度估计的平均绝对误差 (m)
                        MAE_d_v += float((output_v[1] - z_v).abs().sum(0) / batch_size_val * self.z_max)
                        # 累加验证损失
                        loss_v += float(val_loss.item())
                    # 打印训练和验证信息
                    if batch_idx % 10 == 9:
                        print('{} {:d} {:d} \n {:d} {:d} Train Loss: {:.3f} |MAE_R: {:.3f}km MAE_Z: {:.3f}m'.
                              format(model_name, current_epoch, epoch, batch_idx + 1, batch_ndx, loss / (batch_idx + 1),
                                     MAE_r / (batch_idx + 1), MAE_d / (batch_idx + 1)))
                        print('Validation Loss: {:.3f} |MAE_R: {:.3f}km MAE_Z: {:.3f}m\n'.
                              format(loss_v / (batch_idx + 1),
                                     MAE_r_v / (batch_idx + 1), MAE_d_v / (batch_idx + 1)))
                # 释放变量以节省内存
                del dataTrain
                del train_set
                del train_loader
                del val_set
                del val_loader
                del inputs
                del r
                del d
                del train_loss
                del output
                del dataVali
                del inputs_v
                del r_v
                del z_v
                del val_loss
                del output_v
            self.ckp.write_log(
               '{} {:d} {:d} \n {:d} {:d} Train Loss: {:.3f} |MAE_R: {:.3f}km MAE_Z: {:.3f}m'.
                 format(model_name, current_epoch, epoch, batch_idx + 1, batch_ndx, loss / (batch_idx + 1),
                        MAE_r / (batch_idx + 1), MAE_d / (batch_idx + 1)))
            self.ckp.write_log(
                ('Validation Loss: {:.3f} |MAE_R: {:.3f}km MAE_Z: {:.3f}m\n'.
                 format(loss_v / (batch_idx + 1),
                        MAE_r_v / (batch_idx + 1), MAE_d_v / (batch_idx + 1))))
            # Save variables to the list
            log_sigma_of_range.append(log_vars[0])
            log_sigma_of_depth.append(log_vars[1])
            MAPE_of_range.append(100 * MAPE_r / (batch_idx + 1))
            MAPE_of_depth.append(100 * MAPE_d / (batch_idx + 1))
            MAE_of_range.append(MAE_r / (batch_idx + 1))
            MAE_of_depth.append(MAE_d / (batch_idx + 1))
            training_loss.append(loss / (batch_idx + 1))
            MAPE_of_range_v.append(100 * MAPE_r_v / (batch_idx + 1))
            MAPE_of_depth_v.append(100 * MAPE_d_v / (batch_idx + 1))
            MAE_of_range_v.append(MAE_r_v / (batch_idx + 1))
            MAE_of_depth_v.append(MAE_d_v / (batch_idx + 1))
            validation_loss.append(loss_v / (batch_idx + 1))
            # 保存模型信息
            save_info = {  # Saved information
                "current_epoch": current_epoch,  # Number of iterative steps
                "optimizer": optimizer.state_dict(),
                "model": self.model.state_dict(),
                'mini_epoch': mini_epoch,
                'max_epoch': max_epoch,
                'training_loss': training_loss,
                'log_sigma_of_range': log_sigma_of_range,
                'log_sigma_of_depth': log_sigma_of_depth,
                'MAPE_of_range': MAPE_of_range,
                'MAPE_of_depth': MAPE_of_depth,
                'MAE_of_range': MAE_of_range,
                'MAE_of_depth': MAE_of_depth,
                'validation_loss': validation_loss,
                'MAPE_of_range_v': MAPE_of_range_v,
                'MAPE_of_depth_v': MAPE_of_depth_v,
                'MAE_of_range_v': MAE_of_range_v,
                'MAE_of_depth_v': MAE_of_depth_v,
            }
            # 获取保存权重参数文件的文件夹路径
            filepath_weight_para_save_folder = os.path.abspath(self.args.save_file + '/a.weight_parameter')
            # 如果文件夹不存在，则创建
            if not os.path.exists(filepath_weight_para_save_folder):
                os.makedirs(filepath_weight_para_save_folder)
            # 如果保存的权重文件名不为空且不包含'.00'，则删除上一个权重文件（只保留整周期训练权重）
            if save_name_weight_para_file != '' and ('.00' not in save_name_weight_para_file):
                # Delete the last weight file
                os.remove(filepath_weight_para_save_folder + '/' + save_name_weight_para_file + '.pth')
            # 生成保存权重文件的路径
            save_name_weight_para_file = ('{}_{}_epoch_{:.2f}'
                                          .format(model_name, appendix,
                                                  (current_epoch + 1) / mini_epoch))
            save_path_weight_para_file = filepath_weight_para_save_folder + '/' + save_name_weight_para_file + '.pth'
            # 保存模型信息到文件
            torch.save(save_info, save_path_weight_para_file)
            # Release variables
            del log_vars
            del MAPE_d
            del MAPE_r
            del MAE_d
            del MAE_r
            del loss
            del log_vars_v
            del MAPE_d_v
            del MAPE_r_v
            del MAE_d_v
            del MAE_r_v
            del loss_v
            # 让程序暂停指定的休息时间，以便机器降温
            time.sleep(self.args.rest_time)
        # Plot and save 绘图并保存
        # 获取保存训练结果图片的文件夹的绝对路径
        filepath_figure_save_folder = os.path.abspath(self.args.save_file + '/b.training_results')
        # 生成保存训练过程图片的文件名
        save_name_figure_file = save_name_weight_para_file + '_training_process'
        # 生成保存训练过程图片的完整路径
        save_path_figure_file = filepath_figure_save_folder + '/' + save_name_figure_file + '.svg'
        # save_file = './results/training_results/images'
        # save_dict = save_file + '/' + model_path + '.svg'
        # 如果保存图片的文件夹不存在，则创建该文件夹
        if not os.path.exists(filepath_figure_save_folder):
            os.makedirs(filepath_figure_save_folder)
        # 计算距离估计的标准差
        sigma_range = 10 ** np.array(log_sigma_of_range)
        # 计算深度估计的标准差
        sigma_depth = 10 ** np.array(log_sigma_of_depth)
        # 设置图片的分辨率
        dpi = 200
        # 生成 x 轴的数据，代表训练的周期数
        x = torch.arange(0, max_epoch * mini_epoch, 1) / mini_epoch
        # 创建一个新的图形窗口，设置分辨率和尺寸
        pl.figure(dpi=dpi, figsize=(10, 3))
        # 创建一个 1 行 3 列的子图布局，并选择第 1 个子图
        pl.subplot(131)
        # 绘制训练过程中距离估计的平均绝对误差曲线
        bp(x, MAE_of_range, x_tick_off=False, legend_label='Train', color='b')
        # 绘制验证过程中距离估计的平均绝对误差曲线
        bp(x, MAE_of_range_v, title='', x_label='epoch',
           y_label=r'$MAE_r$ (km)', x_tick_off=False, legend_label='Validation', color='r',
           font_size=20, loc='upper right')
        # 选择第 2 个子图
        pl.subplot(132)
        # 绘制训练过程中深度估计的平均绝对误差曲线
        bp(x, MAE_of_depth, x_tick_off=False, legend_label='Train', color='b')
        # 绘制验证过程中深度估计的平均绝对误差曲线
        bp(x, MAE_of_depth_v, title='', x_label='epoch',
           y_label=r'$MAE_d$ (m)', x_tick_off=False, legend_label='Validation', color='r'
           , font_size=20, loc='upper right')
        # 选择第 3 个子图
        pl.subplot(133)
        # 绘制距离估计的标准差曲线
        bp(x, sigma_range, legend_label=r'$\sigma_r$')
        # 绘制深度估计的标准差曲线
        bp(x, sigma_depth, line_style='--', color='r', title='', x_label='epoch',
           y_label=r'$\sigma$', x_tick_off=False, legend_label=r'$\sigma_z$', font_size=20, loc='upper right')
        # 自动调整子图的布局，使它们之间的间距合适
        pl.tight_layout()
        # 将图形保存为 SVG 文件
        pl.savefig(save_path_figure_file, dpi=dpi, bbox_inches='tight')
        # 注释掉的代码，用于显示图形
        # pl.show()
        # save mat data
        # 获取保存训练结果数据的文件夹的绝对路径
        filepath_data_save_folder = os.path.abspath(self.args.save_file + '/b.training_results')
        # 生成保存训练过程数据的文件名
        save_name_data_file = save_name_figure_file
        # 生成保存训练过程数据的完整路径
        save_path_data_file = filepath_data_save_folder + '/' + save_name_data_file + '.mat'
        # save_file = './results/training_results/data'
        # 如果保存数据的文件夹不存在，则创建该文件夹
        if not os.path.exists(filepath_data_save_folder):
            os.makedirs(filepath_data_save_folder)
        # save_dict = save_file + '/' + save_name_weight_para_file + '.mat'
        # 将训练过程中的相关数据保存为 MAT 文件
        sio.savemat(save_path_data_file,
                    {'model_name': model_name,
                     'epoch': x.numpy(), 'MAE_r': np.array(MAE_of_range),
                     'MAE_d': np.array(MAE_of_depth),
                     'MAE_r_val': np.array(MAE_of_range_v),
                     'MAE_d_val': np.array(MAE_of_depth_v),
                     'sigma_range': np.array(sigma_range),
                     'sigma_depth': np.array(sigma_depth),
                     })
        # 打印训练完成的提示信息
        print('------- Finished Training! -------')
        # 返回保存权重参数文件的路径
        return save_path_weight_para_file

    
    def test(self, model_path, mode='test'):
        """
        对模型进行测试的方法。

        :param model_path: 模型的路径，用于加载模型权重。
        :param mode: 测试模式，默认为 'test'。
        :return: 无
        """
        # 获取模型名称
        model_name = self.args.model
        # 写入日志，表示开始评估
        self.ckp.write_log('--------Begin Evaluation-------\n')
        # 选择使用 GPU 或 CPU 进行加速
        default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("running at device %s" % default_device)
        # 设置默认的数据类型
        default_type = torch.float32
        # 获取测试集数据的路径
        path = os.path.abspath(self.args.data_loader_path + '/B.TestSet')

        # 将模型移动到指定设备并设置数据类型
        self.model.to(default_device).type(default_type)
        # 将模型设置为评估模式
        self.model.eval()
        # 打印模型的总参数数量
        print(
            "Total number of parameters in networks is {} ".format(sum(x.numel() for x in self.model.parameters())))

        # 加载模型的检查点
        path_checkpoint = model_path
        checkpoint = torch.load(path_checkpoint)
        # 加载模型的状态字典
        self.model.load_state_dict(checkpoint['model'])  # -*- coding: utf-8 -*-

        # ################ Sensitivity analysis ####################
        # 从配置文件中加载测试集的配置信息
        config = json.loads(open(os.path.join(self.args.data_loader_path, 'config.json'), 'r', encoding='utf-8').read())
        # 获取源距离的上限
        SrU = config['testset']['SourceRange']['UpperLimit']
        # 获取源距离的下限
        SrL = config['testset']['SourceRange']['LowerLimit']
        # 获取源距离的取值数量
        SrN = config['testset']['SourceRange']['NumValues']
        # 获取源深度的上限
        SdU = config['testset']['SourceDepth']['UpperLimit']
        # 获取源深度的下限
        SdL = config['testset']['SourceDepth']['LowerLimit']
        # 获取源深度的取值数量
        SdN = config['testset']['SourceDepth']['NumValues']
        # 生成源距离的数组
        Sr = np.linspace(SrL, SrU, SrN)
        # 生成源深度的数组
        Sd = np.linspace(SdL, SdU, SdN)

        # 计算总数据量
        NDA = len(Sr) * len(Sd)

        # 定义距离估计的相对误差限制
        ErrorLimit_Range = 0.05 * SrU  # Range estimation relative error Limit
        # ErrorLimit_Range = 2  # Range estimation relative error Limit
        # 定义深度估计的绝对误差限制
        ErrorLimit_Depth = 0.05 * SdU  # Depth estimation absolute error limit

        # 设置信噪比
        SNR = 10  # Signal-to-noise ratio
        # sen_para = np.arange(-20, 20, 1)
        # 获取测试集文件的列表
        file_name_all = glob.glob(path + '/*.sim')
        # 生成参数数组
        sen_para = np.arange(0, len(file_name_all), 1)

        # 初始化估计结果数组
        Es = np.zeros([len(sen_para), NDA, 2])  # Estimation results
        # 初始化距离误差数组
        Err_r = np.zeros([len(sen_para), NDA])
        # 初始化深度误差数组
        Err_d = np.zeros([len(sen_para), NDA])
        # 初始化距离预测比例数组
        Proportion_Range_Predict_Array = np.zeros([len(sen_para), 1])
        # 初始化深度预测比例数组
        Proportion_Depth_Predict_Array = np.zeros([len(sen_para), 1])
        # 开始测试
        for i_para in range(len(sen_para)):
            # 设置批次大小
            batch_size = NDA
            # 计算批次数量
            batch_ndx = NDA / batch_size
            # 初始化距离的平均绝对百分比误差
            MAPE_R = 0
            # 初始化深度的平均绝对百分比误差
            MAPE_Z = 0
            # 初始化距离的平均绝对误差
            MAE_R = 0
            # 初始化深度的平均绝对误差
            MAE_Z = 0
            # 创建测试集数据加载器
            test_set = SnSpectrumLoader(file_path=path, length_freq=self.num_frequency, Sr=Sr, Sd=Sd, SNR=SNR,
                                        i_file=i_para, run_mode='test',
                                        model=model_name, num_of_receiver=self.num_receiver)
            dataTest = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

            for iba, mydataTest in enumerate(dataTest):
                with torch.no_grad():
                    # 获取输入数据
                    inputs = mydataTest['C'].float()
                    # 将输入数据转换为 Variable 类型
                    inputs = Variable(inputs)
                    # 将输入数据移动到指定设备并设置数据类型
                    inputs = inputs.to(default_device).type(default_type)

                    # 获取距离标签并归一化
                    r = mydataTest['r'].float() / self.r_max
                    # 将距离标签移动到指定设备并设置数据类型
                    r = r.to(default_device).type(default_type)
                    # 获取深度标签并归一化
                    z = mydataTest['z'].float() / self.z_max
                    # 将深度标签移动到指定设备并设置数据类型
                    z = z.to(default_device).type(default_type)

                    # 前向传播，计算损失、对数方差和输出
                    loss, log_vars, output = self.model(inputs, [r, z])

                    # 计算距离的平均绝对百分比误差
                    MAPE_R += float(((output[0] - r).abs() / r).sum(0) / batch_size)
                    # 计算深度的平均绝对百分比误差
                    MAPE_Z += float(((output[1] - z).abs() / z).sum(0) / batch_size)
                    # 计算距离的平均绝对误差
                    MAE_R += float((output[0] - r).abs().sum(0) / batch_size * self.r_max)
                    # 计算深度的平均绝对误差
                    MAE_Z += float((output[1] - z).abs().sum(0) / batch_size * self.z_max)

                    # 输出测试误差
                    if iba % 5 == 4 or True:
                        print(iba + 1, batch_ndx, 'MAE_R: %.3fkm MAE_Z: %.3fm'
                              % (MAE_R / (iba + 1), MAE_Z / (iba + 1)))
                    # 将输出和标签移动到 CPU 并设置数据类型
                    output[0] = output[0].to(torch.device('cpu')).type(default_type)
                    output[1] = output[1].to(torch.device('cpu')).type(default_type)
                    r = r.to(torch.device('cpu')).type(default_type)
                    z = z.to(torch.device('cpu')).type(default_type)

                    # 保存估计结果
                    Es[i_para, iba * batch_size: (iba + 1) * batch_size, 0] = (
                            output[0] * self.r_max).numpy()
                    Es[i_para, iba * batch_size: (iba + 1) * batch_size, 1] = (
                            output[1] * self.z_max).numpy()

                    # 计算距离误差
                    Err_r[i_para, iba * batch_size: (iba + 1) * batch_size] = np.array((output[0] - r).abs()) *self.r_max
                    # 计算深度误差
                    Err_d[i_para, iba * batch_size: (iba + 1) * batch_size] = np.array((output[1] - z).abs()) *self.z_max
                        # 统计距离预测正确的数量，即距离误差小于误差限制的样本数量
            N_Range_Predict_Right = np.sum(Err_r[i_para, :] < ErrorLimit_Range)
            # 计算距离预测的准确率，并存储到数组中
            Proportion_Range_Predict_Array[i_para] = N_Range_Predict_Right / NDA
            # 打印距离预测的准确率
            print('Proportion rightly of Range Predict: %2.1f%%' % (Proportion_Range_Predict_Array[i_para] * 100))
    
            # 统计深度预测正确的数量，即深度误差小于误差限制的样本数量
            N_Depth_Predict_Right = np.sum(Err_d[i_para, :] < ErrorLimit_Depth)
            # 计算深度预测的准确率，并存储到数组中
            Proportion_Depth_Predict_Array[i_para] = N_Depth_Predict_Right / NDA
            # 打印深度预测的准确率
            print('Proportion rightly of Depth Predict: %2.1f%%' % (Proportion_Depth_Predict_Array[i_para] * 100))
    
        # plot
        # 构建图片保存的文件夹路径
        filepath_figure_save_folder = os.path.abspath(self.args.save_file + '/c.test_results')
        # 构建图片保存的文件名，去除模型路径中的扩展名
        save_name_figure_file = (str.split(model_path, '/')[-1])[:-4] + '_test_accuracy'
        # 构建 SVG 格式图片的保存路径
        save_path_figure_file = filepath_figure_save_folder + '/' + save_name_figure_file + '.svg'
        # 构建 PNG 格式图片的保存路径
        save_path_png_file = filepath_figure_save_folder + '/' + save_name_figure_file + '.png'
        # 如果图片保存的文件夹不存在，则创建该文件夹
        if not os.path.exists(filepath_figure_save_folder):
            os.makedirs(filepath_figure_save_folder)
    
        # 设置 matplotlib 的字体为黑体，用于显示中文
        pl.rcParams['font.sans-serif'] = ['Simhei']
        # 设置图片的分辨率
        dpi = 200
        # 创建一个新的图形窗口
        pl.figure(dpi=dpi, figsize=(10, 4))
        # 创建一个 1 行 2 列的子图布局，并选择第一个子图
        pl.subplot(121)
        # 绘制距离预测准确率的折线图
        bp(sen_para, Proportion_Range_Predict_Array * 100, title='(a) Range estimation results',
           x_label='Parameter', y_label='Proportion (%)', color='b', line_style='-', marker='o')
        # 选择第二个子图
        pl.subplot(122)
        # 绘制深度预测准确率的折线图
        bp(sen_para, Proportion_Depth_Predict_Array * 100, title='(b) Depth estimation results',
           x_label='Parameter', y_label='Proportion (%)', color='b', line_style='-', marker='o')
        # 自动调整子图的布局，使其填充整个图形窗口
        pl.tight_layout()
        # 保存 SVG 格式的图片
        pl.savefig(save_path_figure_file, dpi=dpi, bbox_inches='tight')
        # 保存 PNG 格式的图片
        pl.savefig(save_path_png_file, dpi=dpi, bbox_inches='tight')
        # 关闭图形窗口
        pl.close()
    
        # data
        # 构建数据保存的文件夹路径
        filepath_data_save_folder = os.path.abspath(self.args.save_file + '/c.test_results')
        # 构建数据保存的文件名
        save_name_data_file = save_name_figure_file
        # 构建 MAT 格式数据的保存路径
        save_path_data_file = filepath_data_save_folder + '/' + save_name_data_file + '.mat'
        # 如果数据保存的文件夹不存在，则创建该文件夹
        if not os.path.exists(filepath_data_save_folder):
            os.makedirs(filepath_data_save_folder)
        # 将相关数据保存为 MAT 格式的文件
        sio.savemat(save_path_data_file,
                    {'model_name': model_name,
                     'para': sen_para, 'PRPA': Proportion_Range_Predict_Array,
                     'PDPA': Proportion_Depth_Predict_Array,
                     'SNR': SNR, 'Es': Es, 'Sr': Sr, 'Sd': Sd, 'Err_r': Err_r, 'Err_d': Err_d})