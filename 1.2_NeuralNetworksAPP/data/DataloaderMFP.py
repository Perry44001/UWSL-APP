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
    # Take the upper triangular part of the covariance matrix
    # num_of_receiver = args.num_of_receiver
    upper_tri_real = np.array([])
    upper_tri_image = np.array([])
    for row in range(num_of_receiver):
        upper_row_real = np.arange(row * num_of_receiver + row, (row + 1) * num_of_receiver)
        upper_row_image = np.arange(row * num_of_receiver + row + 1, (row + 1) * num_of_receiver)
        if row % 2 == 1:
            upper_row_real = np.flip(upper_row_real)
            upper_row_image = np.flip(upper_row_image)

        upper_tri_real = np.hstack((upper_tri_real, upper_row_real))
        upper_tri_image = np.hstack((upper_tri_image, upper_row_image))
    upper_tri_real = upper_tri_real.astype(int)
    upper_tri_image = upper_tri_image.astype(int)
    return upper_tri_real, upper_tri_image


def data_load_train_1(file_path, length_freq, SNR_range, num_read_sources, num_of_receiver):
    """
    to randomly load data for training mtl-cnn
        file_path: read file path
        length_freq: length of frequency
        SNR_range: signal-to-noise ratio range
        num_read_sources: number of sources
    """
    # Initialize the matrix
    scm = torch.zeros([num_read_sources, 2 * length_freq, num_of_receiver])
    range_target = torch.zeros([num_read_sources])
    depth_target = torch.zeros([num_read_sources])
    size_data_one_source = 2 * length_freq * num_of_receiver + 2  # The size of the data corresponding to one sound source

    file_name_all = glob.glob(file_path + '/*.sim')  # Read all the file names in the folder

    # Read the files in random order, then read the sound sources in random order
    rand_file_index = randd.randint(0, len(file_name_all), num_read_sources)  # Random order of files
    mask = np.unique(rand_file_index)
    file_path = file_name_all[0]
    file_data_size = os.path.getsize(file_path)
    num_source_one_file = file_data_size // (size_data_one_source * 4)  # Number of sound source points in one file
    tmp = np.zeros(len(file_name_all))
    for v in mask:
        tmp[v] = np.sum(rand_file_index == v)  # Number of sound source points read in the v file
        tp = tmp[v].astype(int)
        RN = np.sort(randd.randint(0, num_source_one_file, tp))  # Take out the reading sound source point sequence
        # in one go

        file_path = file_name_all[v]
        fid = open(file_path, 'rb')
        for iv in range(0, tp):  # Read individual sound sources
            ire = (np.sum(tmp[0: v]) + iv).astype(int)
            SNR = randd.rand(1) * (SNR_range[-1] - SNR_range[0]) + SNR_range[0]
            na = 10 ** (-SNR / 20) / np.sqrt(num_of_receiver)
            # jump = int(RN[iv])
            jump = iv
            na = 0

            fid.seek(jump * size_data_one_source * 4, 0)
            A = (np.array(list(
                struct.unpack('f' * 2 * num_of_receiver * length_freq,
                              fid.read(4 * 2 * num_of_receiver * length_freq))))) \
                .reshape(length_freq, 2 * num_of_receiver)
            pres = A[:, 0:num_of_receiver] + 1j * A[:, num_of_receiver:] + na * (
                    randd.randn(length_freq, num_of_receiver) + 1j * randd.randn(length_freq,
                                                                                 num_of_receiver)) / np.sqrt(2)
            # pres1 = np.expand_dims(pres, axis=1)
            # pres2 = np.expand_dims(pres, axis=2)
            # Rx = (pres1 * pres2.conj())
            scm[ire, 0:length_freq, :] = torch.tensor(np.real(pres))
            scm[ire, length_freq:, :] = torch.tensor(np.imag(pres))
            range_target[ire] = torch.tensor(list(struct.unpack('f' * 1, fid.read(4 * 1))))
            depth_target[ire] = torch.tensor(list(struct.unpack('f' * 1, fid.read(4 * 1))))

    data_read = {'C': scm, 'r': range_target, 'z': depth_target}
    fid.close()
    return data_read


def data_load_test_s_1(file_path, length_freq, Sr, Sd, SNR, i_file=0, num_of_receiver=13):
    """
    to load simulated data at Source range and depth (Sr,Sd) for test
        file_path: read file path
        length_freq: length of frequency
        Sr: Source range vector
        Sd: Source depth vector
        SNR: signal-to-noise ratio
        i_file: index of test file
    """
    test_batch = len(Sr) * len(Sd)
    range_target_vec = np.arange(1, 20.001, 0.1)
    depth_target_vec = np.arange(2, 60.1, 2)  # Simulation range and depth
    size_data_one_source = 2 * length_freq * num_of_receiver + 2  # The size of the data corresponding to one sound source
    size_data_one_depth = size_data_one_source * len(range_target_vec)  # Data size from one depth to the next

    file_name_all = glob.glob(file_path + '/*.sim')  # Read all the file names in the folder
    file_path = file_name_all[i_file]  # Read i-th file

    scm = torch.zeros([test_batch, 2 * length_freq, num_of_receiver])
    range_target = torch.zeros([test_batch])
    depth_target = torch.zeros([test_batch])
    sigma_noise = 10 ** (-SNR / 20) / np.sqrt(num_of_receiver)  # Gaussian noise standard deviation
    fid = open(file_path, 'rb')
    for ire in range(test_batch):
        # Find the location of Sr and Sd in the file
        Sdi = Sd[int(ire / len(Sr))]
        Sri = Sr[ire - int(ire / len(Sr)) * len(Sr)]
        index_d = np.argmin(abs(Sdi - depth_target_vec))
        index_r = np.argmin(abs(Sri - range_target_vec))
        fid.seek((index_d * size_data_one_depth + index_r * size_data_one_source) * 4, 0)

        A = (np.array(list(struct.unpack('f' * 2 * num_of_receiver * length_freq,
                                         fid.read(4 * 2 * num_of_receiver * length_freq))))).reshape(length_freq,
                                                                                                     2 * num_of_receiver)
        pres = A[:, 0:num_of_receiver] + 1j * A[:, num_of_receiver:] + sigma_noise * (
                randd.randn(length_freq, num_of_receiver) + 1j * randd.randn(length_freq, num_of_receiver)) / np.sqrt(2)
        # pres1 = np.expand_dims(pres, axis=1)
        # pres2 = np.expand_dims(pres, axis=2)
        # Rx = (pres1 * pres2.conj())
        # scm[ire, 0:length_freq, :] = torch.tensor(np.real(pres))
        scm[ire, length_freq:, :] = torch.tensor(np.imag(pres))
        range_target[ire] = torch.tensor(list(struct.unpack('f' * 1, fid.read(4 * 1))))
        depth_target[ire] = torch.tensor(list(struct.unpack('f' * 1, fid.read(4 * 1))))

    data_read = {'C': scm, 'r': range_target, 'z': depth_target}
    fid.close()
    return data_read


def data_load_test_m_1(file_path, length_freq, num_of_receiver):
    """
    to load data of measured data for inference
        file_path: read file path
        length_freq: length of frequency
    """
    file_data_size = os.path.getsize(file_path)
    size_data_one_source = 2 * length_freq * num_of_receiver + 2  # The size of the data corresponding to one sound source
    test_m_batch = file_data_size // (size_data_one_source * 4)  # Number of sound sources in a file
    scm = torch.zeros([test_m_batch, 2 * length_freq, num_of_receiver])
    range_target = torch.zeros([test_m_batch])
    depth_target = torch.zeros([test_m_batch])
    fid = open(file_path, 'rb')
    for ire in range(test_m_batch):
        A = (np.array(list(struct.unpack('f' * 2 * num_of_receiver * length_freq,
                                         fid.read(4 * 2 * num_of_receiver * length_freq))))). \
            reshape(length_freq, 2 * num_of_receiver)
        pres = A[:, 0:num_of_receiver] + 1j * A[:, num_of_receiver:]
        # pres1 = np.expand_dims(pres, axis=1)
        # pres2 = np.expand_dims(pres, axis=2)
        # Rx = (pres1 * pres2.conj())
        scm[ire, 0:length_freq, :] = torch.tensor(np.real(pres))
        scm[ire, length_freq:, :] = torch.tensor(np.imag(pres))
        range_target[ire] = torch.tensor(list(struct.unpack('f' * 1, fid.read(4 * 1))))
        depth_target[ire] = torch.tensor(list(struct.unpack('f' * 1, fid.read(4 * 1))))
    data_read = {'C': scm, 'r': range_target, 'z': depth_target}
    fid.close()
    return data_read


def data_load_train_2(file_path, length_freq, SNR_range, num_read_sources, num_of_receiver):
    """
    to randomly load data for training mtl-cnn
        file_path: read file path
        length_freq: length of frequency
        SNR_range: signal-to-noise ratio range
        num_read_sources: number of sources
    """
    upper_tri_real, upper_tri_image = upper_tri(num_of_receiver)
    scm = torch.zeros([num_read_sources, 1, 1 * length_freq, num_of_receiver * num_of_receiver])
    range_target = torch.zeros([num_read_sources])
    depth_target = torch.zeros([num_read_sources])
    file_name_all = glob.glob(file_path + '/*.sim')

    rand_file_index = randd.randint(0, len(file_name_all), num_read_sources)
    mask = np.unique(rand_file_index)
    file_path = file_name_all[0]
    file_data_size = os.path.getsize(file_path)
    size_data_one_source = 2 * length_freq * num_of_receiver + 2  # The size of the data corresponding to one sound source
    num_source_one_file = int(file_data_size // (size_data_one_source * 4))
    tmp = np.zeros(len(file_name_all))
    for v in mask:
        tmp[v] = np.sum(rand_file_index == v)
        tp = tmp[v].astype(int)
        RN = np.sort(randd.randint(0, num_source_one_file, tp))
        file_path = file_name_all[v]
        fid = open(file_path, 'rb')
        for iv in range(0, tp):
            ire = (np.sum(tmp[0: v]) + iv).astype(int)
            SNR = randd.rand(1) * (SNR_range[-1] - SNR_range[0]) + SNR_range[0]
            na = 10 ** (-SNR / 20) / np.sqrt(num_of_receiver)
            Jump = int(RN[iv])

            fid.seek(Jump * size_data_one_source * 4, 0)

            A = (np.array(list(struct.unpack('f' * 2 * num_of_receiver * length_freq,
                                             fid.read(4 * 2 * num_of_receiver * length_freq))))). \
                reshape(length_freq, 2 * num_of_receiver)
            pres = A[:, 0:num_of_receiver] + 1j * A[:, num_of_receiver:] + na * (
                    randd.randn(length_freq, num_of_receiver) + 1j * randd.randn(length_freq,
                                                                                 num_of_receiver)) / np.sqrt(2)
            pres1 = np.expand_dims(pres, axis=2)
            pres2 = np.expand_dims(pres, axis=1)
            Rx = (pres1 * pres2.conj())
            IMAGE = np.hstack(
                ((np.real(Rx).reshape(length_freq, -1))[:, upper_tri_real],
                 (np.imag(Rx).reshape(length_freq, -1))[:, upper_tri_image]))
            IMAGE_nor = (IMAGE - IMAGE.mean()) / IMAGE.std()

            scm[ire, 0, :, :] = torch.tensor(IMAGE_nor)
            range_target[ire] = torch.tensor(list(struct.unpack('f' * 1, fid.read(4 * 1))))
            depth_target[ire] = torch.tensor(list(struct.unpack('f' * 1, fid.read(4 * 1))))

    data_read = {'C': scm, 'r': range_target, 'z': depth_target}
    fid.close()
    return data_read


def data_load_test_s_2(file_path, length_freq, Sr, Sd, SNR, ifile=0, num_of_receiver=13):
    """
    to load simulated data at Source range and depth (Sr,Sd) for test
        file_path: read file path
        length_freq: length of frequency
        Sr: Source range vector
        Sd: Source depth vector
        SNR: signal-to-noise ratio
        i_file: index of test file
    """
    upper_tri_real, upper_tri_image = upper_tri(num_of_receiver)
    test_batch = len(Sr) * len(Sd)
    # range_target_vec = np.arange(1, 20.001, 0.25)
    # depth_target_vec = np.arange(5, 100.1, 5)
    range_target_vec = np.arange(1, 20.001, 0.1)
    depth_target_vec = np.arange(2, 60.1, 2)  # Simulation range and depth
    size_data_one_source = 2 * length_freq * num_of_receiver + 2  # The size of the data corresponding to one sound source
    size_data_one_depth = size_data_one_source * len(range_target_vec)

    scm = torch.zeros([test_batch, 1, 1 * length_freq, num_of_receiver * num_of_receiver])
    range_target = torch.zeros([test_batch])
    depth_target = torch.zeros([test_batch])
    sigma_noise = 10 ** (-SNR / 20) / np.sqrt(num_of_receiver)

    file_name_all = glob.glob(file_path + '/*.sim')
    file_path = file_name_all[ifile]
    fid = open(file_path, 'rb')

    for ire in range(test_batch):
        Sdi = Sd[int(ire / len(Sr))]
        Sri = Sr[ire - int(ire / len(Sr)) * len(Sr)]
        index_d = np.argmin(abs(Sdi - depth_target_vec))
        index_r = np.argmin(abs(Sri - range_target_vec))
        fid.seek((index_d * size_data_one_depth + index_r * size_data_one_source) * 4, 0)

        A = (np.array(
            list(struct.unpack('f' * 2 * num_of_receiver * length_freq,
                               fid.read(4 * 2 * num_of_receiver * length_freq))))).reshape(length_freq,
                                                                                           2 * num_of_receiver)
        pres = A[:, 0:num_of_receiver] + 1j * A[:, num_of_receiver:] + sigma_noise * (
                randd.randn(length_freq, num_of_receiver) + 1j * randd.randn(length_freq, num_of_receiver)) / np.sqrt(2)
        pres1 = np.expand_dims(pres, axis=2)
        pres2 = np.expand_dims(pres, axis=1)
        Rx = (pres1 * pres2.conj())

        IMAGE = np.hstack(
            ((np.real(Rx).reshape(length_freq, -1))[:, upper_tri_real],
             (np.imag(Rx).reshape(length_freq, -1))[:, upper_tri_image]))
        IMAGE_nor = (IMAGE - IMAGE.mean()) / IMAGE.std()

        scm[ire, 0, :, :] = torch.tensor(IMAGE_nor)
        range_target[ire] = torch.tensor(list(struct.unpack('f' * 1, fid.read(4 * 1))))
        depth_target[ire] = torch.tensor(list(struct.unpack('f' * 1, fid.read(4 * 1))))
    data_read = {'C': scm, 'r': range_target, 'z': depth_target}
    fid.close()
    return data_read


def data_load_test_m_2(file_path, length_freq, num_of_receiver):
    """
    to load data of measured data for feature-compressed CNNs (MTL-UNET, XCEPTION) inference
        file_path: read file path
        length_freq: length of frequency
    """
    upper_tri_real, upper_tri_image = upper_tri(num_of_receiver)
    file_data_size = os.path.getsize(file_path)
    size_data_one_source = 2 * length_freq * num_of_receiver + 2  # The size of the data corresponding to one sound source
    test_m_batch = file_data_size // (size_data_one_source * 4)

    scm = torch.zeros([test_m_batch, 1, 1 * length_freq, num_of_receiver * num_of_receiver])
    range_target = torch.zeros([test_m_batch])
    depth_target = torch.zeros([test_m_batch])
    fid = open(file_path, 'rb')
    for ire in range(test_m_batch):
        A = (np.array(list(struct.unpack('f' * 2 * num_of_receiver * length_freq,
                                         fid.read(4 * 2 * num_of_receiver * length_freq))))). \
            reshape(length_freq, 2 * num_of_receiver)
        pres = A[:, 0:num_of_receiver] + 1j * A[:, num_of_receiver:]
        pres1 = np.expand_dims(pres, axis=2)
        pres2 = np.expand_dims(pres, axis=1)
        Rx = (pres1 * pres2.conj())

        image = np.hstack(((np.real(Rx).reshape(length_freq, -1))[:, upper_tri_real],
                           (np.imag(Rx).reshape(length_freq, -1))[:, upper_tri_image]))
        image_nor = (image - image.mean()) / image.std()

        scm[ire, 0, :, :] = torch.tensor(image_nor)
        range_target[ire] = torch.tensor(list(struct.unpack('f' * 1, fid.read(4 * 1))))
        depth_target[ire] = torch.tensor(list(struct.unpack('f' * 1, fid.read(4 * 1))))
    data_read = {'C': scm, 'r': range_target, 'z': depth_target}
    fid.close()
    return data_read


class SnSpectrumLoader(data.Dataset):
    def __init__(self, file_path='', length_freq=151, num_of_receiver=13, SNR_range=[15, 15], num_read_sources=32,
                 Sr=[10], Sd=[10],
                 SNR=15, i_file=0, run_mode='train', model='mtl_cnn'):
        super(SnSpectrumLoader, self).__init__()
        if model == 'mtl_cnn':
            if run_mode == 'train':
                self.dataT = data_load_train_1(file_path, length_freq=length_freq, SNR_range=SNR_range,
                                               num_read_sources=num_read_sources, num_of_receiver=num_of_receiver)
            elif run_mode == 'test':
                self.dataT = data_load_test_s_1(file_path, length_freq=length_freq, Sr=Sr, Sd=Sd, SNR=SNR,
                                                i_file=i_file, num_of_receiver=num_of_receiver)
            else:
                self.dataT = data_load_test_m_1(file_path, length_freq=length_freq, num_of_receiver=num_of_receiver)
        else:
            if run_mode == 'train':
                self.dataT = data_load_train_2(file_path, length_freq=length_freq, SNR_range=SNR_range,
                                               num_read_sources=num_read_sources, num_of_receiver=num_of_receiver)
            elif run_mode == 'test':
                self.dataT = data_load_test_s_2(file_path, length_freq=length_freq, Sr=Sr, Sd=Sd, SNR=SNR,
                                                ifile=i_file, num_of_receiver=num_of_receiver)
            else:
                self.dataT = data_load_test_m_2(file_path, length_freq=length_freq, num_of_receiver=num_of_receiver)

    def __getitem__(self, index):
        C = self.dataT['C'][index]
        r = self.dataT['r'][index]
        z = self.dataT['z'][index]
        data = {'C': C, 'r': r, 'z': z}
        return data

    def __len__(self):
        return len(self.dataT['r'])


if __name__ == '__main__':
    from option_data_loader import args_d
    import time
    import pylab as pl
    import numpy as np

    t1 = time.time()
    data_path = '../../1.1 Dataset Simulation/1.1.3 Dataset/'

    dataset = SnSpectrumLoader(file_path=data_path + args_d.file_path, length_freq=args_d.length_freq,
                               SNR_range=args_d.SNR_range, num_read_sources=args_d.num_read_sources, Sr=np.array([10]),
                               Sd=np.array([10]), SNR=args_d.SNR, i_file=args_d.i_file, run_mode=args_d.run_mode,
                               model=args_d.model)
    print(len(dataset))
    dataTrain = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    print(len(dataTrain))
    for i, data in enumerate(dataTrain):
        C, r, z = data['C'], data['r'], data['z']  # eg: label -> ('0',) ('1',) It is a tuple.
        print('dataset', C.size(), r[0])  # type( num is tensor and label[0] is str.)
        if i == 1:
            break
        t2 = time.time()
        print('Program run time: %fs' % (t2 - t1))
        print(r, z)
        aa = C[0, 0, :, :].squeeze()
        fig = pl.figure(dpi=200, figsize=(5, 4))
        h = pl.pcolor(aa, cmap=pl.cm.get_cmap('jet'))
        ax = pl.gca()
        ax.invert_yaxis()
        fig.colorbar(h)
        pl.show()
