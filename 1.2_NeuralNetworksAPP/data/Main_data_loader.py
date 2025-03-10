from option_data_loader import args_d
import time
from DataloaderV3 import SnSpectrumLoader
import torch.utils.data as data
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
    print('程序运行时间%fs' % (t2 - t1))
    print(r, z)
    aa = C[0, 0, :, :].squeeze()
    # bb = aa/aa.max()
    # pl.pcolor(bb,cmap='gray')
    fig = pl.figure(dpi=200, figsize=(5, 4))
    h = pl.pcolor(aa, cmap=pl.cm.get_cmap('jet'))  # , vmin=-2, vmax=6)
    ax = pl.gca()
    ax.invert_yaxis()
    fig.colorbar(h)
    pl.show()
