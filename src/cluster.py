# -*- coding:utf-8 -*-


import os
import sys

import numpy as np 
# import scipy.spatial.distance as sciDist
# import sklearn.preprocessing as sklPrep
import sklearn.cluster as sklCluster
import sklearn.neighbors as sklNeighbors
from sklearn.externals import joblib


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


def process_line_from_text():
    pass


def read_stip_file(path=None, linedict=None, fact=1, mode='all '): 
    # global total_1, total_2
    stips = []
    for count, line in enumerate(read_line_from_text(path=path)):
        # the first 3 lines are infos about the stip file and will be discarded.
        if count-3 >= 0:
            # sampleing
            # if (linedict == total_2).any():
            sline = line.strip().split()
            # print(sline)

            try:
                # map(float, sline)
                [float(s) for s in sline]
            except Exception:
                print(" ValueError: could not convert string to float: ", sline)
                pass
            else:
                # fline = map(float, sline)
                fline = [float(s) for s in sline[7:]]
                stips += [fline]
                # total_1 += 1
            # total_2 += 1
        else:
            # print(line.strip().split())
            pass

    '''Be careful of empty list!!! len(stips) for cline'''
    # print(len(stips), total_1, total_2)
    # print(len(stips), len(stips[0]))
    # print(len(stips))
    return len(stips), stips


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


def generate_kmeans_model(cates=None, round=None, flag=None, K=None): 
    '''generating kMmeas model offline'''
    x_rand = np.load('../data/stips_r%d_f%d.npy'%(round,flag))

    kms = sklCluster.KMeans(n_clusters=K, n_jobs=-1, random_state=0)
    kms.fit(x_rand)

    joblib.dump(kms, '../data/kmeans_r%d_f%d_k%d.model'%(round, flag, K), compress=3)
    # kms = joblib.load(kms, '../data/kmeans_r%d_f%d_k%d.model'%(round, flag, K))

    np.save('../data/kmeans_r%d_f%d_k%d.model'%(round, flag, K), kms.cluster_centers_)
    # centroids = np.load('../data/kmeans_r%d_f%d_k%d.model'%(round, flag, K)) 


def apply_kmeans_model(cates=None, round=None, flag=None, K=None, C=None):
    '''generating cline, label and bag-of-features, which can be feed to classifier or regressor directly'''
    # joblib.dump(kms, '../data/kmeans_r%d_f%d_k%d.model'%(round, flag, K), compress=3)
    # kms = joblib.load('../data/kmeans_r%d_f%d_k%d.model'%(round, flag, K))

    # np.save('../data/kmeans_r%d_f%d_k%d.model'%(round, flag, K), kms.cluster_centers_)
    centroids = np.load('../data/kmeans_r%d_f%s_k%d.npy'%(round, '1', K))

    # round level
    # flag: {0：not used, 1:train, 2:test}

    # clinedict = {
    #     1:5613856,
    #     2:5483247,
    #     3:5367763
    # }

    # random.seed(a=round)
    # linedict = np.array(random.sample([i for i in range(clinedict[round])], 100000))
    # print(linedict[:7])
    
    bovfs = np.zeros((0,K)) # bag-of-visual-features
    cline = np.zeros((0,1))
    label = np.zeros((0,C))

    for j in range(len(cates)):
        # cate level
        print(j+1, cates[j])
        print('%s/%s_test_split%d.txt'%(splitdir, cates[j], round))

        # read split file
        # c0,c1,c2 = 0,0,0
        for line in read_line_from_text(path='%s/%s_test_split%d.txt'%(splitdir, cates[j], round)):
            # sample level
            sline = line.strip().split()
            vname, mask = sline[0], sline[1]

            if mask == flag:
                # c,s = read_stip_file(path='%s/%s/%s.txt'%(stipdir, cates[j], vname)) # version 1
                s = np.loadtxt('%s/%s/%s.txt'%(stipdir, cates[j], vname), comments='#')[:,7:] # version 2, however, much slower
                c = s.shape[0]

                # cline processing
                c = np.array([c]).reshape((-1, 1))

                # label processing
                l = np.zeros((1,C)) # one-zero label
                l[0,j] = 1

                # predicting
                hist = np.zeros((1,K))
                if len(s) == 0:
                    pass 
                else: 
                    # index = kms.predict(s)
                    index = knn_search(s, centroids)
                    print(index.shape, index[:10,:].T)
                    hist = np.histogram(index, K, range = (0, K))[0].reshape((-1,K))
                    print(hist.shape, hist[0,:10])                

                cline = np.vstack((cline, c))
                label = np.vstack((label, l))
                bovfs = np.vstack((bovfs, hist))
                print(cline.shape, label.shape, bovfs.shape)

            else:
                pass
            # print('\n')

            # ''' debug '''
            # if sline[1] == '0': c0 +=1
            # elif sline[1] == '1' : c1 += 1
            # else: c2 += 1

            # break

        # print(c0,c1,c2)

        # break

    # to reduce the memory load
    bovfs = np.array(bovfs)
    label = np.array(label)
    cline = np.array(cline).reshape(-1,1)

    print([cline.shape, label.shape, bovfs.shape])

    np.save('../data/bovfs_r%d_f%s_k%d'%(round, flag, K), bovfs)
    np.save('../data/label_r%d_f%s_k%d'%(round, flag, K), label)
    np.save('../data/cline_r%d_f%s_k%d'%(round, flag, K), cline) 


if __name__ == '__main__':
    splitdir = '../testTrainMulti_7030_splits'
    stipdir = '../hmdb51_org_stips'

    cates = os.listdir(stipdir)
    # print(len(cates))
    # read_stip_file(path='../data/brush_hair/Blonde_being_brushed_brush_hair_f_nm_np2_ri_med_0.avi.txt')

    # flag: {0：not used, 1:train, 2:test}
    # round = 1
    # flag = '1'
    # K = 4000

    round = int(sys.argv[1])
    flag = sys.argv[2]
    K = int(sys.argv[3])
    C = 51
    
    # generate_kmeans_model(cates=cates, round=round, flag=flag, K=K)
    apply_kmeans_model(cates=cates, round=round, flag=flag, K=K, C=C)