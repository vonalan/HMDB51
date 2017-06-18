# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
from sklearn.metrics import mean_squared_error as sklmse

import rbfnn
import stsm
import scale


# scale to [-1,1]
def auto_normalize(dataset, bias=0.0, scale=1.0, mode=None): 
    xbias = bias
    xscale = scale
    
    if mode == 'train': 
        xmin = dataset.min() 
        xmax = dataset.max() 
        xbias = (xmax + xmin)/2.0 
        xscale = xmax - bias
    elif mode == 'valid': 
        pass 
    else: 
        raise Exception('Value Error! ')

    dataset = (dataset - xbias)/xscale

    return dataset, xbias, xscale


# api
def calc_err(y_true, y_predict):
    return sklmse(y_true, y_predict, multioutput='True')


def calc_acc(y_true, y_predict):
    tidx = np.argmax(y_true, axis=1)
    pidx = np.argmax(y_predict, axis=1)

    eidx = (tidx==pidx)
    acc = eidx.sum()/eidx.shape[0]

    return acc

def load_data(): 
    '''cline, label, bovfs'''
    pass 

def normalization(): 
    '''necessary'''
    pass 

if __name__ == '__main__':
    c_train, y_train, x_train = load_data()
    x_train, bias, scale = scale.auto_normalize()

    rbfnnC = rbfnn.RBFNN(indim=x_train.shape[1], numCenter=90, outdim=y_train.shape[1])
    rbfnn.fit(x_train, y_train)
    out_train = rbfnn.predict(x_train)

    c_testa, y_testa, x_testa = load_data()
    x_testa, _, _ = scale.auto_normalize()
    out_testa = rbfnn.predict(x_testa)

    # evaluation
    acc_train = calc_acc(y_train, out_train)
    err_train = calc_err(y_train, out_train)
    acc_testa = calc_acc(y_testa, out_testa)
    err_testa = calc_err(y_testa, out_testa)

    stsm_train = stsm.calc_stsm(x_train, y_train, rbfnnC.W, rbfnnC.U, rbfnnC.V)
    lge_train = np.power(np.sqrt(err_train) + np.sqrt(stsm_train),2)