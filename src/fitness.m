function [ O ] = fitness(S)
    % calculating fitness for population; 

    resDim = 4;
    O = zeros(size(S,1), resDim); 
    
    centroids_path = ''; 
    centroids = import(centroids_path);

    % matlabpool local 4; 
    parfor i  = 1 : size(S,1)
        fprintf('calculating fitness for individual %d of %d ... \n',i, size(S,1)); 

        % mask 
        s = S(i,:); 
        mcentroids = centroids(s==1,:); 
        k = size(mcentroids,1); 
        
        % flag = {0:not used, 1:train, 2:testa}; 
        [cline_1, label_1, bovfs_1] = kmeans_transform(cates, round, 1, K, C, centroids); 
        [cline_2, label_2, bovfs_2] = kmeans_transform(cates, round, 2, K, C, centroids); 

        % train and validate classifier
        [mse, stsm, trainAcc, testAcc] = run_rbfnn(bovfs_1, bovfs_2, label_1, label_2); 
        
        O(i,:) = [mse, stsm, trainAcc, testAcc]; 
    end; 

    % matlabpool close; 

end


function [cline, label, bovfs] = kmeans_transform(cates, round, flag, K, C, centroids)
    cline = zeros((0,1)); 
    label = zeros((0,C)); 
    bovfs = zeros((0,K)); 

    for j = 1 : size(cates,1)
        print('%s/%s_test_split%d.txt'%(splitdir, cates[j], round)); 

        split_path = '%s/%s_test_split%d.txt'%(splitdir, cates[j], round); 
        split_set = import(split_path); 

        for k = 1 : size(split_set)
            [vname, mask] = [split_set(k,1), split_set(k,2)]

            if mask == flag
                stip_path = '%s/%s/%s.txt'%(stipdir, cates[j], vname); 
                [c,s] = read_stip_file(stip_path); 

                s= knnsearch(centroids, s); 
                % s = build_hist(c, s, k); 
                s = tabulate(s); 

                cline = [cline;c]; 
                label = [label;c];
                bovfs = [bovfs;s];
            end; 
        end; 
    end; 
    print(size(cline), size(label), size(bovfs)); 
end; 


% bug bug bug; 
function stips = read_stip_file(stip_path)
    stips = import(stip_path)(4:,:); 
end; 