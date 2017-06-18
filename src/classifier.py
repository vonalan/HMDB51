# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
from sklearn.metrics import mean_squared_error as sklmse

import rbfnn
import stsm
import scale


def load_data(round=None, flag=None, K=None): 
    '''cline, label, bovfs'''
    cline = np.load('../data/cline_r%d_f%s_k%d.npy'%(round, flag, K))
    label = np.load('../data/label_r%d_f%s_k%d.npy'%(round, flag, K))
    bofvs = np.load('../data/bofvs_r%d_f%s_k%d.npy'%(round, flag, K))
    
    return cline, label, bofvs
    # pass 


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


if __name__ == '__main__':
    round = 1 
    # flag = '1'
    K = 4000
    M = 90
    C = 51

    rbfnnC = rbfnn.RBFNN(indim=K, numCenter=M, outdim=C, alpha=1.0)

    c_train, y_train, x_train = load_data(round=round, flag='1', K=K)
    x_train, bias, scale = auto_normalize(x_train, bias=0.0, scale=1.0, mode='train')
    rbfnnC.fit(x_train, y_train)

    c_testa, y_testa, x_testa = load_data(round=round, flag='1', K=K)
    x_testa, _, _ = auto_normalize(x_testa, bias=bias, scale=scale, mode='valid')

    out_train = rbfnnC.predict(x_train)
    out_testa = rbfnnC.predict(x_testa)
    
    acc_train = calc_acc(y_train, out_train)
    err_train = calc_err(y_train, out_train)
    stsm_train = stsm.calc_stsm(x_train, y_train, rbfnnC.W, rbfnnC.U, rbfnnC.V, Q=0.1)
    lgem_train = np.power(np.sqrt(err_train) + np.sqrt(stsm_train),2)

    acc_testa = calc_acc(y_testa, out_testa)
    err_testa = calc_err(y_testa, out_testa)
    # stsm_testa = stsm.calc_stsm(x_testa, y_testa, rbfnnC.W, rbfnnC.U, rbfnnC.V, Q=0.1)
    # lgem_testa = np.power(np.sqrt(err_testa) + np.sqrt(stsm_testa),2)

    print(acc_train, err_train, acc_testa, err_testa, lgem_train)