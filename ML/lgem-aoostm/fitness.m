function [ O ] = fitness(S)
    % calculating fitness for population; 
    
    global cline_train label_train stips_train cline_testa label_testa stips_testa centroids;
    
    m = size(S,1);
    resDim = 4;
    O = zeros(size(S,1), resDim); 
    
    % matlabpool local 4; 
    for i = 1 : m
    % parfor i  = 1 : m
        % mask 
        s = S(i,:); 
        mcentroids = centroids(s==1,:); 

        % flag = {0:not used, 1:train, 2:testa}; 
        % for loop
        [x_train, x_testa] = cluster(mcentroids); 
        % parfor loop 
        % [x_train, x_testa] = cluster_parfor(cline_train, stips_train, cline_testa, stips_testa, centroids)
        
        % train and validate classifier
        [param] = classifier_light(cline_train, label_train, x_train, cline_testa, label_testa, x_testa); 
        [err_train, stsm_train, acc_train, acc_testa] = reduce_results(param); 
        
        fprintf('Fitness value for %d of %d: {k:%d, acc_train: %.4f, acc_testa: %.4f, err_train: %.4f, stsm_train: %.4f}. \n', ...
           i, m, size(mcentroids,1), acc_train, acc_testa, err_train, stsm_train); 
        
        O(i,:) = [err_train, stsm_train, acc_train, acc_testa]; 
        % break;
    end; 
    % matlabpool close; 
end


function [err_train, stsm_train, acc_train, acc_testa] = reduce_results(param)
    rbfnnC = param.rbfnnC; 
    
    acc_train = rbfnnC.acc_train;
    acc_testa = rbfnnC.acc_testa;
    err_train = mean(rbfnnC.err_train);
    % err_testa = mean(rbfnnC.err_testa);
    stsm_train = mean(rbfnnC.stsm_train);
    % lgem_train = mean(rbfnnC.lgem_train);
end 