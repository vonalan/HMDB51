# -*-coding:utf-8 -*-

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
    for line in read_line_from_text(path=path): 
        pass

    return stips



cates = ['brush_hair', 'dive', 'shake_hands']

for i in range(1):
    for cts in cates: 
        
        # read split file 
        # c0,c1,c2 = 0,0,0
        for line in read_line_from_text(path='../data/%s_test_split%d.txt'%(cts, i+1)): 
            # sample level
            sline = line.strip().split()

            # {0：not used, 1:train, 2:test}
            if sline[1] == '1': 
                stip_train += read_stip_file()
                y_train += [] 
            else: 
                stip_testa += read_stip_file()
                y_testa += [] 
            
            # ''' debug '''
            # if sline[1] == '0': c0 +=1 
            # elif sline[1] == '1' : c1 += 1
            # else: c2 += 1

        # print(c0,c1,c2)
        

        # break

    # cluster() 

    # build_hist()

    # classify() 

