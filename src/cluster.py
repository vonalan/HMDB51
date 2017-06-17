# -*- coding:utf-8 -*-


import os
import sys

import numpy as np 
# import scipy.spatial.distance as sciDist
# import sklearn.preprocessing as sklPrep
import sklearn.cluster as sklCluster
import sklearn.neighbors as sklNeighbors
from sklearn.externals import joblib


def kMeans(dataSet=None, k=None):
    kms = sklCluster.KMeans(n_clusters=k, n_jobs=1, random_state=0)
    kms.fit(dataSet)

    return kms


# an equivalent implemetation of kMeans.predict()
def knn_search(dataset, centroids): 
    nbs = sklNeighbors.NearestNeighbors(n_neighbors=1)
    nbs.fit(centroids)
    knn = nbs.kneighbors(dataset, n_neighbors=1,return_distance=False) 
    # idx = knn.flatten()
    
    return knn


def build_hist(dataset, bins=None): 
    return np.histogram(dataset, bins, range = (0, bins))[0]


if __name__ == '__main__':
    # flag: {0：not used, 1:train, 2:test}
    round = 1 
    flag = 1
    x_rand = np.load('../data/stips_r%d_f%d.npy'%(round,flag))

    K = 128
    kms = sklCluster.KMeans(n_clusters=K, n_jobs=-1, random_state=0).fit(x_rand)

    joblib.dump(kms, '../data/kmeans_r%d_f%d_k%d.model'%(round, flag, K), compress=3)
    # kms = joblib.load(kms, '../data/kmeans_r%d_f%d_k%d.model'%(round, flag, K))

    np.save('../data/kmeans_r%d_f%d_k%d.model'%(round, flag, K), kms.cluster_centers_)
    # centroids = np.load('../data/kmeans_r%d_f%d_k%d.model'%(round, flag, K))