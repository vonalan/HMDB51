# -*- coding:utf-8 -*-


import os 
import sys 
import datetime 
import platform

import numpy as np


def read_line_from_text(path=None): 
    # path = utilities.getPath(path)
    rf = open(path, 'r')

    while True:
        line = rf.readline()
        if line:
            yield line
        else:
            break

    rf.close()


def read_block_from_text(path=None): 
    rf = open(path, 'r')
    lines = rf.readlines()
    return lines


def get_current_time(): 
    return str(datetime.datetime.now()) + ' '


print(get_current_time() + 'Implementing the application in the platform of %s'%platform.system())


if __name__ == '__main__': 
    lines = read_block_from_text(path='../data/brush_hair_test_split1.txt')
    print(len(lines))
    linesM = np.loadtxt('../data/brushing_hair_brush_hair_f_nm_np2_ba_goo_2.avi.txt',comments='#')[:,7:]
    print(linesM.shape, linesM[:10,:4])