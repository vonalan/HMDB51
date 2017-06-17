# -*- coding:utf-8 -*-


import os 
import sys 
import datetime 
import platform


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


def get_current_time(): 
    return str(datetime.datetime.now()) + ' '


print(get_current_time() + 'Implementing the application in the platform of %s'%platform.system())