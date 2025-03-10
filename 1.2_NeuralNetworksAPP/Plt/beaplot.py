# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:52:30 2021

@author: admin
该程序用来使plot更漂亮
"""

import pylab as pl
import numpy as np


def plot(x, y, line_style='-', color='b', title='', x_label='', y_label='', x_tick_off=False,
         legend_label='', marker=',', font_size=14, loc='lower left'):
    pl.rcParams['font.sans-serif'] = ['Simhei']
    pl.rcParams['axes.unicode_minus'] = False
    pl.plot(x, y, color=color, linestyle=line_style, label=legend_label, marker=marker, markersize=2)
    lab_size = font_size - 2
    pl.title(title, fontsize=font_size)
    pl.ylabel(y_label, fontsize=lab_size)
    if legend_label != '':
        pl.legend(loc=loc, fontsize=lab_size)
    if x_tick_off:
        pl.xticks([])
    else:
        pl.xticks(fontsize=lab_size)
        pl.xlabel(x_label, fontsize=lab_size)
    pl.yticks(fontsize=lab_size)

    if __name__ == '__main__':
        import beaplot
        y = list([1, 2, 3, 1, 2, 3])
        x = np.arange(0, len(y), 1)
        beaplot(x, y, 'test', '1', '2', True, '3', loc='upper left')