# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 16:59:17 2021

@author: admin
读取实验数据
"""

import torch
import torch.utils.data as data
import struct
import numpy as np
import numpy.random as randd
import os
import glob
import sys


def upper_tri(num_of_receiver):
    """
    提取协方差矩阵的上三角部分

    参数:
    num_of_receiver (int): 接收器的数量

    返回:
    tuple: 包含两个数组，分别为上三角实部索引和上三角虚部索引
    """
    # 初始化上三角实部索引数组
    upper_tri_real = np.array([])
    # 初始化上三角虚部索引数组
    upper_tri_image = np.array([])
    # 遍历每一行
    for row in range(num_of_receiver):
        # 计算当前行上三角实部的索引范围
        upper_row_real = np.arange(row * num_of_receiver + row, (row + 1) * num_of_receiver)
        # 计算当前行上三角虚部的索引范围
        upper_row_image = np.arange(row * num_of_receiver + row + 1, (row + 1) * num_of_receiver)
        # 如果当前行索引为奇数
        if row % 2 == 1:
            # 反转上三角实部索引
            upper_row_real = np.flip(upper_row_real)
            # 反转上三角虚部索引
            upper_row_image = np.flip(upper_row_image)
        # 将当前行的上三角实部索引添加到总的上三角实部索引数组中
        upper_tri_real = np.hstack((upper_tri_real, upper_row_real))
        # 将当前行的上三角虚部索引添加到总的上三角虚部索引数组中
        upper_tri_image = np.hstack((upper_tri_image, upper_row_image))
    # 将上三角实部索引数组转换为整数类型
    upper_tri_real = upper_tri_real.astype(int)
    # 将上三角虚部索引数组转换为整数类型
    upper_tri_image = upper_tri_image.astype(int)
    # 返回上三角实部索引和上三角虚部索引
    return upper_tri_real, upper_tri_image

def data_load_train_1(file_path, length_freq, SNR_range, num_read_sources, num_of_receiver):
    """
    随机加载数据用于训练 mtl-cnn 模型
    参数:
        file_path (str): 读取文件的路径
        length_freq (int): 频率的长度
        SNR_range (list): 信噪比范围
        num_read_sources (int): 要读取的声源数量
        num_of_receiver (int): 接收器的数量
    返回:
        dict: 包含协方差矩阵、距离目标和深度目标的字典
    """
    # 初始化矩阵
    # 初始化协方差矩阵，形状为 (num_read_sources, 2 * length_freq, num_of_receiver, num_of_receiver)
    scm = torch.zeros([num_read_sources, 2 * length_freq, num_of_receiver, num_of_receiver])
    # 初始化距离目标向量
    range_target = torch.zeros([num_read_sources])
    # 初始化深度目标向量
    depth_target = torch.zeros([num_read_sources])
    # 单个声源对应的数据大小
    size_data_one_source = 2 * length_freq * num_of_receiver + 2  
    # 获取文件夹中所有 .sim 文件的文件名
    file_name_all = glob.glob(file_path + '/*.sim')  
    # 随机选择文件顺序，然后随机选择每个文件中的声源顺序
    # 随机选择文件索引
    rand_file_index = randd.randint(0, len(file_name_all), num_read_sources)  
    # 获取唯一的文件索引
    mask = np.unique(rand_file_index)
    # 先取第一个文件路径
    file_path = file_name_all[0]
    # 获取文件的大小
    file_data_size = os.path.getsize(file_path)
    # 计算一个文件中的声源点数
    num_source_one_file = file_data_size // (size_data_one_source * 4)  
    # 记录每个文件中要读取的声源点数
    tmp = np.zeros(len(file_name_all))
    for v in mask:
        # 计算第 v 个文件中要读取的声源点数
        tmp[v] = np.sum(rand_file_index == v)  
        # 转换为整数类型
        tp = tmp[v].astype(int)
        # 随机选择该文件中的声源点序列
        RN = np.sort(randd.randint(0, num_source_one_file, tp))  # 逐个读取声源  in one go
        # 获取第 v 个文件的路径
        file_path = file_name_all[v]
        # 以二进制读取模式打开文件
        fid = open(file_path, 'rb')
        for iv in range(0, tp):  # 逐个读取声源
            # 计算当前声源在总数据中的索引
            ire = (np.sum(tmp[0: v]) + iv).astype(int)
            # 随机生成一个信噪比
            SNR = randd.rand(1) * (SNR_range[-1] - SNR_range[0]) + SNR_range[0]
            # 计算噪声标准差
            na = 10 ** (-SNR / 20) / np.sqrt(num_of_receiver)
            # 计算文件读取的偏移量
            jump = int(RN[iv])
            # 移动文件指针到指定位置
            fid.seek(jump * size_data_one_source * 4, 0)
            # 读取二进制数据并转换为 numpy 数组
            A = (np.array(list(
                struct.unpack('f' * 2 * num_of_receiver * length_freq, fid.read(4 * 2 * num_of_receiver * length_freq))))) \
                .reshape(length_freq, 2 * num_of_receiver)
            # 计算压力值，添加噪声
            pres = A[:, 0:num_of_receiver] + 1j * A[:, num_of_receiver:] + na * (
                    randd.randn(length_freq, num_of_receiver) + 1j * randd.randn(length_freq, num_of_receiver)) / np.sqrt(2)
            # 扩展维度
            pres1 = np.expand_dims(pres, axis=1)
            # 扩展维度
            pres2 = np.expand_dims(pres, axis=2)
            # 计算协方差矩阵
            Rx = (pres1 * pres2.conj())
            # 存储协方差矩阵的实部
            scm[ire, 0:length_freq, :, :] = torch.tensor(np.real(Rx))
            # 存储协方差矩阵的虚部
            scm[ire, length_freq:, :, :] = torch.tensor(np.imag(Rx))
            # 读取距离目标值
            range_target[ire] = torch.tensor(list(struct.unpack('f' * 1, fid.read(4 * 1))))
            # 读取深度目标值
            depth_target[ire] = torch.tensor(list(struct.unpack('f' * 1, fid.read(4 * 1))))
    # 封装数据为字典
    data_read = {'C': scm, 'r': range_target, 'z': depth_target}
    # 关闭文件
    fid.close()
    return data_read

def data_load_test_s_1(file_path, length_freq, Sr, Sd, SNR, i_file=0, num_of_receiver=13):
    """
    加载指定源距离和深度 (Sr, Sd) 的模拟数据用于测试

    参数:
        file_path (str): 读取文件的路径
        length_freq (int): 频率的长度
        Sr (list): 源距离向量
        Sd (list): 源深度向量
        SNR (float): 信噪比
        i_file (int, 可选): 测试文件的索引，默认为 0
        num_of_receiver (int, 可选): 接收器的数量，默认为 13
    返回:
        dict: 包含协方差矩阵、距离目标和深度目标的字典
    """
    # 计算测试批次的大小，即源距离和源深度组合的数量
    test_batch = len(Sr) * len(Sd)
    # 设定模拟的距离范围
    range_target_vec = Sr
    # 设定模拟的深度范围
    depth_target_vec = Sd
    # 计算单个声源对应的数据大小
    size_data_one_source = 2 * length_freq * num_of_receiver + 2
    # 计算从一个深度到下一个深度的数据大小
    size_data_one_depth = size_data_one_source * len(range_target_vec)

    # 获取指定路径下所有 .sim 文件的文件名
    file_name_all = glob.glob(file_path + '/*.sim')
    # 选择第 i_file 个文件进行读取
    file_path = file_name_all[i_file]

    # 初始化协方差矩阵，形状为 (test_batch, 2 * length_freq, num_of_receiver, num_of_receiver)
    scm = torch.zeros([test_batch, 2 * length_freq, num_of_receiver, num_of_receiver])
    # 初始化距离目标向量
    range_target = torch.zeros([test_batch])
    # 初始化深度目标向量
    depth_target = torch.zeros([test_batch])
    # 计算高斯噪声的标准差
    sigma_noise = 10 ** (-SNR / 20) / np.sqrt(num_of_receiver)
    # 以二进制读取模式打开文件
    fid = open(file_path, 'rb')
    # 遍历每个测试批次
    for ire in range(test_batch):
        # 找到当前批次对应的源深度
        Sdi = Sd[int(ire / len(Sr))]
        # 找到当前批次对应的源距离
        Sri = Sr[ire - int(ire / len(Sr)) * len(Sr)]
        # 找到当前源深度在深度目标向量中的索引
        index_d = np.argmin(abs(Sdi - depth_target_vec))
        # 找到当前源距离在距离目标向量中的索引
        index_r = np.argmin(abs(Sri - range_target_vec))
        # 将文件指针移动到指定位置
        fid.seek((index_d * size_data_one_depth + index_r * size_data_one_source) * 4, 0)

        # 读取二进制数据并转换为 numpy 数组
        A = (np.array(list(struct.unpack('f' * 2 * num_of_receiver * length_freq,
                                         fid.read(4 * 2 * num_of_receiver * length_freq))))).reshape(length_freq,
                                                                                                  2 * num_of_receiver)
        # 计算压力值，添加噪声
        pres = A[:, 0:num_of_receiver] + 1j * A[:, num_of_receiver:] + sigma_noise * (
                randd.randn(length_freq, num_of_receiver) + 1j * randd.randn(length_freq, num_of_receiver)) / np.sqrt(2)
        # 扩展维度
        pres1 = np.expand_dims(pres, axis=1)
        # 扩展维度
        pres2 = np.expand_dims(pres, axis=2)
        # 计算协方差矩阵
        Rx = (pres1 * pres2.conj())
        # 存储协方差矩阵的实部
        scm[ire, 0:length_freq, :, :] = torch.tensor(np.real(Rx))
        # 存储协方差矩阵的虚部
        scm[ire, length_freq:, :, :] = torch.tensor(np.imag(Rx))
        # 读取距离目标值
        range_target[ire] = torch.tensor(list(struct.unpack('f' * 1, fid.read(4 * 1))))
        # 读取深度目标值
        depth_target[ire] = torch.tensor(list(struct.unpack('f' * 1, fid.read(4 * 1))))

    # 封装数据为字典
    data_read = {'C': scm, 'r': range_target, 'z': depth_target}
    # 关闭文件
    fid.close()
    return data_read

def data_load_train_2(file_path, length_freq, SNR_range, num_read_sources, num_of_receiver):
    """
    随机加载数据用于训练 mtl-cnn 模型

    参数:
        file_path (str): 读取文件的路径
        length_freq (int): 频率的长度
        SNR_range (list): 信噪比范围
        num_read_sources (int): 要读取的声源数量
        num_of_receiver (int): 接收器的数量

    返回:
        dict: 包含协方差矩阵、距离目标和深度目标的字典
    """
    # 调用 upper_tri 函数获取协方差矩阵上三角部分的实部和虚部索引
    upper_tri_real, upper_tri_image = upper_tri(num_of_receiver)
    # 初始化协方差矩阵，形状为 (num_read_sources, 1, length_freq, num_of_receiver * num_of_receiver)
    scm = torch.zeros([num_read_sources, 1, 1 * length_freq, num_of_receiver * num_of_receiver])
    # 初始化距离目标向量
    range_target = torch.zeros([num_read_sources])
    # 初始化深度目标向量
    depth_target = torch.zeros([num_read_sources])
    # 获取指定路径下所有 .sim 文件的文件名
    file_name_all = glob.glob(file_path + '/*.sim')

    # 随机选择文件索引
    rand_file_index = randd.randint(0, len(file_name_all), num_read_sources)
    # 获取唯一的文件索引
    mask = np.unique(rand_file_index)
    # 先取第一个文件路径
    file_path = file_name_all[0]
    # 获取文件的大小
    file_data_size = os.path.getsize(file_path)
    # 计算单个声源对应的数据大小
    size_data_one_source = 2 * length_freq * num_of_receiver + 2  # 单个声源对应的数据大小
    # 计算一个文件中的声源点数
    num_source_one_file = int(file_data_size // (size_data_one_source * 4))
    # 记录每个文件中要读取的声源点数
    tmp = np.zeros(len(file_name_all))
    # 遍历每个唯一的文件索引
    for v in mask:
        # 计算第 v 个文件中要读取的声源点数
        tmp[v] = np.sum(rand_file_index == v)
        # 转换为整数类型
        tp = tmp[v].astype(int)
        # 随机选择该文件中的声源点序列
        RN = np.sort(randd.randint(0, num_source_one_file, tp))
        # 获取第 v 个文件的路径
        file_path = file_name_all[v]
        # 以二进制读取模式打开文件
        fid = open(file_path, 'rb')
        # 遍历每个要读取的声源
        for iv in range(0, tp):
            # 计算当前声源在总数据中的索引
            ire = (np.sum(tmp[0: v]) + iv).astype(int)
            # 随机生成一个信噪比
            SNR = randd.rand(1) * (SNR_range[-1] - SNR_range[0]) + SNR_range[0]
            # 计算噪声标准差
            na = 10 ** (-SNR / 20) / np.sqrt(num_of_receiver)
            # 计算文件读取的偏移量
            Jump = int(RN[iv])

            # 移动文件指针到指定位置
            fid.seek(Jump * size_data_one_source * 4, 0)

            # 读取二进制数据并转换为 numpy 数组
            A = (np.array(list(struct.unpack('f' * 2 * num_of_receiver * length_freq,
                                             fid.read(4 * 2 * num_of_receiver * length_freq))))). \
                reshape(length_freq, 2 * num_of_receiver)
            # 计算压力值，添加噪声
            pres = A[:, 0:num_of_receiver] + 1j * A[:, num_of_receiver:] + na * (
                    randd.randn(length_freq, num_of_receiver) + 1j * randd.randn(length_freq, num_of_receiver)) / np.sqrt(2)
            # 扩展维度
            pres1 = np.expand_dims(pres, axis=2)
            # 扩展维度
            pres2 = np.expand_dims(pres, axis=1)
            # 计算协方差矩阵
            Rx = (pres1 * pres2.conj())
            # 提取协方差矩阵上三角部分的实部和虚部，并拼接成一个矩阵
            IMAGE = np.hstack(
                ((np.real(Rx).reshape(length_freq, -1))[:, upper_tri_real],
                 (np.imag(Rx).reshape(length_freq, -1))[:, upper_tri_image]))
            # 对矩阵进行归一化处理
            IMAGE_nor = (IMAGE - IMAGE.mean()) / IMAGE.std()

            # 存储归一化后的矩阵到协方差矩阵中
            scm[ire, 0, :, :] = torch.tensor(IMAGE_nor)
            # 读取距离目标值
            range_target[ire] = torch.tensor(list(struct.unpack('f' * 1, fid.read(4 * 1))))
            # 读取深度目标值
            depth_target[ire] = torch.tensor(list(struct.unpack('f' * 1, fid.read(4 * 1))))

    # 封装数据为字典
    data_read = {'C': scm, 'r': range_target, 'z': depth_target}
    # 关闭文件
    fid.close()
    return data_read

def data_load_test_s_2(file_path, length_freq, Sr, Sd, SNR, ifile=0, num_of_receiver=13):
    """
    加载指定源距离和深度 (Sr, Sd) 的模拟数据用于测试

    参数:
        file_path (str): 读取文件的路径
        length_freq (int): 频率的长度
        Sr (list): 源距离向量
        Sd (list): 源深度向量
        SNR (float): 信噪比
        ifile (int, 可选): 测试文件的索引，默认为 0
        num_of_receiver (int, 可选): 接收器的数量，默认为 13

    返回:
        dict: 包含协方差矩阵、距离目标和深度目标的字典
    """
    # 调用 upper_tri 函数获取协方差矩阵上三角部分的实部和虚部索引
    upper_tri_real, upper_tri_image = upper_tri(num_of_receiver)
    # 计算测试批次的大小，即源距离和源深度组合的数量
    test_batch = len(Sr) * len(Sd)
    # 设定模拟的距离范围
    range_target_vec = Sr
    # 设定模拟的深度范围
    depth_target_vec = Sd
    # 计算单个声源对应的数据大小
    size_data_one_source = 2 * length_freq * num_of_receiver + 2
    # 计算从一个深度到下一个深度的数据大小
    size_data_one_depth = size_data_one_source * len(range_target_vec)

    # 初始化协方差矩阵，形状为 (test_batch, 1, length_freq, num_of_receiver * num_of_receiver)
    scm = torch.zeros([test_batch, 1, 1 * length_freq, num_of_receiver * num_of_receiver])
    # 初始化距离目标向量
    range_target = torch.zeros([test_batch])
    # 初始化深度目标向量
    depth_target = torch.zeros([test_batch])
    # 计算高斯噪声的标准差
    sigma_noise = 10 ** (-SNR / 20) / np.sqrt(num_of_receiver)

    # 获取指定路径下所有 .sim 文件的文件名
    file_name_all = glob.glob(file_path + '/*.sim')
    # 选择第 ifile 个文件进行读取
    file_path = file_name_all[ifile]
    # 以二进制读取模式打开文件
    fid = open(file_path, 'rb')

    # 遍历每个测试批次
    for ire in range(test_batch):
        # 找到当前批次对应的源深度
        Sdi = Sd[int(ire / len(Sr))]
        # 找到当前批次对应的源距离
        Sri = Sr[ire - int(ire / len(Sr)) * len(Sr)]
        # 找到当前源深度在深度目标向量中的索引
        index_d = np.argmin(abs(Sdi - depth_target_vec))
        # 找到当前源距离在距离目标向量中的索引
        index_r = np.argmin(abs(Sri - range_target_vec))
        # 将文件指针移动到指定位置
        fid.seek((index_d * size_data_one_depth + index_r * size_data_one_source) * 4, 0)

        # 读取二进制数据并转换为 numpy 数组
        A = (np.array(
            list(struct.unpack('f' * 2 * num_of_receiver * length_freq,
                               fid.read(4 * 2 * num_of_receiver * length_freq))))).reshape(length_freq,
                                                                                        2 * num_of_receiver)
        # 计算压力值，添加噪声
        pres = A[:, 0:num_of_receiver] + 1j * A[:, num_of_receiver:] + sigma_noise * (
                randd.randn(length_freq, num_of_receiver) + 1j * randd.randn(length_freq, num_of_receiver)) / np.sqrt(2)
        # 扩展维度
        pres1 = np.expand_dims(pres, axis=2)
        # 扩展维度
        pres2 = np.expand_dims(pres, axis=1)
        # 计算协方差矩阵
        Rx = (pres1 * pres2.conj())

        # 提取协方差矩阵上三角部分的实部和虚部，并拼接成一个矩阵
        IMAGE = np.hstack(
            ((np.real(Rx).reshape(length_freq, -1))[:, upper_tri_real],
             (np.imag(Rx).reshape(length_freq, -1))[:, upper_tri_image]))
        # 对矩阵进行归一化处理
        IMAGE_nor = (IMAGE - IMAGE.mean()) / IMAGE.std()

        # 存储归一化后的矩阵到协方差矩阵中
        scm[ire, 0, :, :] = torch.tensor(IMAGE_nor)
        # 读取距离目标值
        range_target[ire] = torch.tensor(list(struct.unpack('f' * 1, fid.read(4 * 1))))
        # 读取深度目标值
        depth_target[ire] = torch.tensor(list(struct.unpack('f' * 1, fid.read(4 * 1))))

    # 封装数据为字典
    data_read = {'C': scm, 'r': range_target, 'z': depth_target}
    # 关闭文件
    fid.close()
    return data_read

class SnSpectrumLoader(data.Dataset):
    def __init__(self, file_path='', length_freq=151, num_of_receiver=13, SNR_range=[15, 15], num_read_sources=32,
                 Sr=[10], Sd=[10],
                 SNR=15, i_file=0, run_mode='train', model='mtl_cnn'):
        """
        初始化 SnSpectrumLoader 类
    
        参数:
            file_path (str): 数据文件的路径，默认为空字符串
            length_freq (int): 频率的长度，默认为 151
            num_of_receiver (int): 接收器的数量，默认为 13
            SNR_range (list): 信噪比范围，默认为 [15, 15]
            num_read_sources (int): 要读取的声源数量，默认为 32
            Sr (list): 源距离向量，默认为 [10]
            Sd (list): 源深度向量，默认为 [10]
            SNR (float): 信噪比，默认为 15
            i_file (int): 测试文件的索引，默认为 0
            run_mode (str): 运行模式，'train' 表示训练模式，'test' 表示测试模式，默认为 'train'
            model (str): 模型名称，默认为 'mtl_cnn'
        """
        # 调用父类的构造函数
        super(SnSpectrumLoader, self).__init__()
        # 根据不同的模型选择不同的数据加载方式
        if model == 'mtl_cnn':
            # 如果是训练模式
            if run_mode == 'train':
                # 调用 data_load_train_1 函数加载训练数据
                self.dataT = data_load_train_1(file_path, length_freq=length_freq, SNR_range=SNR_range,
                                               num_read_sources=num_read_sources, num_of_receiver=num_of_receiver)
            # 如果是测试模式
            elif run_mode == 'test':
                # 调用 data_load_test_s_1 函数加载测试数据
                self.dataT = data_load_test_s_1(file_path, length_freq=length_freq, Sr=Sr, Sd=Sd, SNR=SNR,
                                                i_file=i_file, num_of_receiver=num_of_receiver)
        else:
            # 如果是训练模式
            if run_mode == 'train':
                # 调用 data_load_train_2 函数加载训练数据
                self.dataT = data_load_train_2(file_path, length_freq=length_freq, SNR_range=SNR_range,
                                               num_read_sources=num_read_sources, num_of_receiver=num_of_receiver)
            # 如果是测试模式
            elif run_mode == 'test':
                # 调用 data_load_test_s_2 函数加载测试数据
                self.dataT = data_load_test_s_2(file_path, length_freq=length_freq, Sr=Sr, Sd=Sd, SNR=SNR,
                                                ifile=i_file, num_of_receiver=num_of_receiver)
    def __getitem__(self, index):
        """
        根据给定的索引获取数据集中的一个样本

        参数:
            index (int): 要获取的样本的索引

        返回:
            dict: 包含协方差矩阵、距离目标和深度目标的字典
        """
        # 从数据集中获取指定索引的协方差矩阵
        C = self.dataT['C'][index]
        # 从数据集中获取指定索引的距离目标
        r = self.dataT['r'][index]
        # 从数据集中获取指定索引的深度目标
        z = self.dataT['z'][index]
        # 将协方差矩阵、距离目标和深度目标封装到一个字典中
        data = {'C': C, 'r': r, 'z': z}
        # 返回封装好的数据字典
        return data
    def __len__(self):
        """
        返回数据集的样本数量

        返回:
            int: 数据集的样本数量，通过距离目标向量的长度来确定
        """
        return len(self.dataT['r'])

if __name__ == '__main__':
    """
    主程序入口，用于测试数据加载和可视化
    """
    # 从 option_data_loader 模块导入 args_d 对象
    from option_data_loader import args_d
    # 导入时间模块，用于记录程序运行时间
    import time
    # 导入 pylab 模块，用于绘图
    import pylab as pl
    # 导入 numpy 模块，用于数值计算
    import numpy as np
    # 导入 json 模块，用于处理 JSON 数据
    import json

    # 记录程序开始时间
    t1 = time.time()
    # 定义数据文件路径
    data_path = r"E:\2.0_UWSL_Datasets\1200m\A.TrainingSet"
    # 获取数据文件所在的父目录路径
    parent_path = os.path.dirname(data_path)

    # 打开并读取配置文件
    config = json.loads(open(os.path.join(parent_path, 'config.json'), 'r', encoding='utf-8').read())
    # 根据配置文件中的接收器深度数量更新 args_d 对象的 num_of_receiver 属性
    args_d.num_of_receiver = len(config['ReceiverDepth'])
    # 根据配置文件中的频率数量更新 args_d 对象的 length_freq 属性
    args_d.length_freq = config['Frequency']['NumValues']
    # 打印接收器数量
    print('num_of_receiver:', args_d.num_of_receiver)
    # 打印频率长度
    print('length_freq:', args_d.length_freq)

    # 创建 SnSpectrumLoader 数据集对象
    dataset = SnSpectrumLoader(file_path=data_path, length_freq=args_d.length_freq,  num_of_receiver=args_d.num_of_receiver,
                               SNR_range=args_d.SNR_range, num_read_sources=args_d.num_read_sources, Sr=np.array([10]),
                               Sd=np.array([10]), SNR=args_d.SNR, i_file=args_d.i_file, run_mode=args_d.run_mode,
                               model=args_d.model)
    # 打印数据集的长度
    print(len(dataset))
    # 创建数据加载器对象
    dataTrain = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    # 打印数据加载器的长度
    print(len(dataTrain))
    # 遍历数据加载器
    for i, data in enumerate(dataTrain):
        # 从数据中提取协方差矩阵、距离目标和深度目标
        C, r, z = data['C'], data['r'], data['z']  # eg: label -> ('0',) ('1',) It is a tuple.
        # 打印协方差矩阵的大小
        print('size of Convmatrix: ', C.size())  # type( num is tensor and label[0] is str.)
        # 打印距离目标
        print('r:', r)
        # 打印深度目标
        print('z:', z)
        # 如果已经处理了 1 个批次的数据，则跳出循环
        if i == 1:
            break
        # 记录当前时间
        t2 = time.time()
        # 打印程序运行时间
        print('Program run time: %fs' % (t2 - t1))
        # 打印距离目标和深度目标
        print(r, z)
        # 提取协方差矩阵的第一个元素并压缩维度
        aa = C[0, 0, :, :].squeeze()
        # 创建一个新的图形对象
        fig = pl.figure(dpi=200, figsize=(5, 4))
        # 绘制伪彩色图
        h = pl.pcolor(aa, cmap=pl.cm.get_cmap('jet'))
        # 获取当前坐标轴对象
        ax = pl.gca()
        # 反转 y 轴
        ax.invert_yaxis()
        # 添加颜色条
        fig.colorbar(h)
        # 显示图形
        pl.show()
