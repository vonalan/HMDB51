# -*-coding:utf-8 -*-

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


def read_stip_file(path=None): 
    stips = []
    for count, line in enumerate(read_line_from_text(path=path)):
        # the first 3 lines are infos about the stip file and will be discarded.
        if count-3 >=0:
            sline = line.strip().split()
            # print(sline)

            try:
                # map(float, sline)
                [float(s) for s in sline]
            except Exception:
                # print(" ValueError: could not convert string to float: ", sline)
                pass
            else:
                # fline = map(float, sline)
                fline = [float(s) for s in sline[7:]]
                stips += [fline]
        else:
            print(line.strip().split())

    print(len(stips), len(stips[0]))
    return len(stips), stips



cates = ['brush_hair', 'dive', 'shake_hands']


def aggragate_stip_file(round=None, flag=None):
    # round level
    # flag: {0：not used, 1:train, 2:test}

    stips = []
    cline = []
    label = []

    for j in range(len(cates)):
        # cate level
        # read split file
        # c0,c1,c2 = 0,0,0
        for line in read_line_from_text(path='../data/%s_test_split%d.txt'%(cates[j], round)):
            # sample level
            sline = line.strip().split()
            vname, mask = sline[0], sline[1]

            if mask == flag:
                c,s = read_stip_file(path='../data/%s/%s.txt'%(cates[j], vname))
                cline += [c]
                stips += s
                label += [j]
            else:
                pass

            # ''' debug '''
            # if sline[1] == '0': c0 +=1
            # elif sline[1] == '1' : c1 += 1
            # else: c2 += 1

            break

        # print(c0,c1,c2)

        break

    stipsM = np.array(stips)
    labelM = np.array(label).reshape(-1,1)
    clineM = np.array(cline).reshape(-1,1)

    print(clineM.shape, labelM.shape, stipsM.shape)

    np.save('./stips_r%d_f%s.txt'%(round, flag), stipsM)
    np.save('./label_r%d_f%s.txt'%(round, flag), labelM)
    np.save('./cline_r%d_f%s.txt'%(round, flag), clineM)

if __name__ == '__main__': 
    # read_stip_file(path='../data/brush_hair/Blonde_being_brushed_brush_hair_f_nm_np2_ri_med_0.avi.txt')
    aggragate_stip_file(round=1, flag='1')