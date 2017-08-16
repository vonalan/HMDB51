function [C] = preload( round, K, ns1, ns2, nsr)
    global cline_train label_train stips_train cline_testa label_testa stips_testa centroids;
    
    cates_dir = '../hmdb51_org_stips';
    % cates_raw = dir(cates_dir);
    cates_raw = ls(cates_dir);
    cates = cates_raw(3:end,:); 
    C = size(cates,1);   
    
    % path_cline_train = ['../data/cline_train_r',int2str(round),'_f1', '_s', int2str(ns1), '.txt'];
    % path_cline_testa = ['../data/cline_testa_r',int2str(round),'_f2', '_s', int2str(ns2), '.txt'];
    % path_label_train = ['../data/label_train_r',int2str(round),'_f1', '_s', int2str(ns1), '.txt'];
    % path_label_testa = ['../data/label_train_r',int2str(round),'_f2', '_s', int2str(ns2), '.txt'];
    % path_stips_train = ['../data/stips_train_r',int2str(round),'_f1', '_s', int2str(ns1), '.txt'];
    % path_stips_testa = ['../data/stips_testa_r',int2str(round),'_f2', '_s', int2str(ns2), '.txt'];
    
    path_centroids = ['../data/common/stips_centroids_r',int2str(round),'_f1', '_s', int2str(nsr), '_k', int2str(K), '.mat'];
    path_train = ['../data/common/cline_label_stips_r',int2str(round),'_f1', '_s', int2str(ns1), '.mat'];
    path_testa = ['../data/common/cline_label_stips_r',int2str(round),'_f2', '_s', int2str(ns2), '.mat'];
    
    % struct;
    centroids = load(path_centroids);
    data_train = load(path_train);
    data_testa = load(path_testa);
    
    % matrix; 
    centroids = centroids.centroids;
    cline_train = data_train.cline; 
    cline_testa = data_testa.cline;
    stips_train = [data_train.stips_part1; data_train.stips_part2; data_train.stips_part3;];
    stips_testa = data_testa.stips_part1; 
    
    prelb_train = data_train.label+1;
    prelb_testa = data_testa.label+1; 
    
    label_train = zeros(size(prelb_train,1),C);
    for i = 1 : size(prelb_train,1)
        label_train(i,prelb_train(i,1))=1;
    end;
    
    label_testa = zeros(size(prelb_testa,1),C);
    for i = 1 : size(prelb_testa,1)
        label_testa(i,prelb_testa(i,1))=1;
    end;
    
end

