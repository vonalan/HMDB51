# -*-coding:utf-8 -*-

import os 
import sys
import random

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


def read_stip_file(path=None, linedict=None, fact=1): 
    global total 
    stips = []
    for count, line in enumerate(read_line_from_text(path=path)):
        # the first 3 lines are infos about the stip file and will be discarded.
        if count-3 >= 0:
            # sampleing
            if (linedict == total).any():
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
            total += 1
        else:
            # print(line.strip().split())
            pass

    '''Be careful of empty list!!!'''
    print(len(stips))
    return len(stips), stips


def aggragate_stip_file(round=None, flag=None):
    # round level
    # flag: {0：not used, 1:train, 2:test}

    clinedict = {
        1:5613856,
        2:5483247,
        3:5367763
    }

    random.seed(a=round)
    linedict = np.array(random.sample([i for i in range(clinedict[round])], 100000))
    print(linedict[:7])
    
    stips = []
    cline = []
    label = []

    for j in range(len(cates)):
        # cate level
        print(j+1, cates[j])

        # read split file
        # c0,c1,c2 = 0,0,0
        for line in read_line_from_text(path='%s/%s_test_split%d.txt'%(splitdir, cates[j], round)):
            # sample level
            sline = line.strip().split()
            vname, mask = sline[0], sline[1]

            if mask == flag:
                c,s = read_stip_file(path='%s/%s/%s.txt'%(stipdir, cates[j], vname), linedict=linedict)
                cline += [c]
                stips += s
                label += [j]
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
    stips = np.array(stips)
    label = np.array(label).reshape(-1,1)
    cline = np.array(cline).reshape(-1,1)

    print([cline.shape, label.shape, stips.shape])

    np.save('../data/stips_r%d_f%s'%(round, flag), stips)
    np.save('../data/label_r%d_f%s'%(round, flag), label)
    np.save('../data/cline_r%d_f%s'%(round, flag), cline)


if __name__ == '__main__': 
    splitdir = '../testTrainMulti_7030_splits'
    stipdir = '../hmdb51_org_stips'

    cates = os.listdir(stipdir)
    # print(len(cates))
    # read_stip_file(path='../data/brush_hair/Blonde_being_brushed_brush_hair_f_nm_np2_ri_med_0.avi.txt')

    total = 0

    # flag: {0：not used, 1:train, 2:test}
    aggragate_stip_file(round=1, flag='1')
    # # aggragate_stip_file(round=2, flag='1')
    # aggragate_stip_file(round=3, flag='1')
    