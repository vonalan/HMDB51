function [ O ] = fitness_light(S)
    % calculating fitness for population; 
    
    resDim = 4;
    O = zeros(size(S,1), resDim); 
    
    cates_dir = '../hmdb51_org_stips';
    % cates_raw = dir(cates_dir);
    cates_raw = ls(cates_dir);
    cates = cates_raw(3:end,:); 
    
    round = 1; 
    flag = 1; 
    C = size(cates,1); 
    K = 128; 
    
    centroids_path = ['../data/centroids_r', int2str(round), '_f', int2str(flag), '_k', int2str(K), '.txt']; 
    fprintf(centroids_path);
    fprintf('\n');
    centroids = importdata(centroids_path);
    
    % global c_train y_train x_train c_testa y_testa x_testa;
    % matlabpool local 4; 
    for i  = 1 : size(S,1)
        fprintf('calculating fitness for individual %d of %d... \n',i, size(S,1)); 
        
        % mask 
        s = S(i,:); 
        mcentroids = centroids(s==1,:); 

        % flag = {0:not used, 1:train, 2:testa}; 
        [c_train, y_train, x_train] = cluster_light(cates, round, 1, C, mcentroids); 
        [c_testa, y_testa, x_testa] = cluster_light(cates, round, 2, C, mcentroids); 
%         x_train_path = '../data/x_train.txt'; 
%         x_testa_path = '../data/x_testa.txt'; 
%         y_train_path = '../data/y_train.txt'; 
%         y_testa_path = '../data/y_testa.txt'; 
% 
%         x_train = importdata(x_train_path);
%         x_testa = importdata(x_testa_path); 
%         y_train = importdata(y_train_path); 
%         y_testa = importdata(y_testa_path); 
%         
%         x_train = x_train(:,60:end); 
%         x_testa = x_testa(:,60:end);
        
        
        % train and validate classifier
        [param] = classifier_light(c_train, y_train, x_train, c_testa, y_testa, x_testa); 
        [err_train, stsm_train, acc_train, acc_testa] = reduce_results(param); 
        
        fprintf('acc_train: %.4f, acc_testa: %.4f, err_train: %.4f, err_testa: %.4f\n', ...
           acc_train, acc_testa, err_train, stsm_train); 
        
        O(i,:) = [err_train, stsm_train, acc_train, acc_testa]; 
        %��break;
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