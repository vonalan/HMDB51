function [ O ] = fitness(S)
    % calculating fitness for population; 

    resDim = 4;
    O = zeros(size(S,1), resDim); 
    
    centroids_path = ''; 
    centroids = import(centroids_path); 

    % matlabpool local 4; 
    parfor i  = 1 : size(S,1)
        fprintf('calculating fitness for individual %d of %d ... \n',i, size(S,1)); 
        s = S(i,:); 
        mcentroids = centroids(s==1,:); 
        k = size(mcentroids,1); 
        
        [cline, label, bovfs] = read_samples(round, flag, k)
        
        [mse, stsm, trainAcc, testAcc] = rbfnn(hist_train, hist_test, y_train, y_test); 

        
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


function stips = read_stip_file(stip_path)
    stips = import(stip_path)(4:,:); 
end; 


function x_bow  = build_hist(cdata, xdata, nbins)
    x_bow = zeros(size(cdata, 1), nbins); 
    
    cbegin = 1; 
    % cend = 1; 
    for i = 1 : size(cdata, 1)        
        cend = cbegin + cdata(i,:) - 1; 
        x_temp = xdata(cbegin:cend, :); 
        hist = tabulate(x_temp)'; 
        % x_bow(i,1:size(hist,2)) = hist(3,:); 
        x_bow(i,1:size(hist,2)) = hist(2,:); 
        
        cbegin = cend + 1; 
    end; 
end

