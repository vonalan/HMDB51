function [ O ] = fitness(S)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    % global c_train c_test x_train x_test centroids y_train y_test; 
    
    resDim = 4;
    O = zeros(size(S,1), resDim); 
    
    % matlabpool local 4; 
    parfor i  = 1 : size(S,1)
        fprintf('calculating fitness for individual %d of %d ... \n',i, size(S,1)); 
        s = S(i,:); 
        mcentroids = centroids(s==1,:); 
        nbins = size(mcentroids,1); 
        
        [cline, label, bovfs] = read_samples(round, flag, k)
        
        [mse, stsm, trainAcc, testAcc] = rbfnn(hist_train, hist_test, y_train, y_test); 

        
        O(i,:) = [mse, stsm, trainAcc, testAcc]; 
    end; 

    % matlabpool close; 

end

function [cline, label, bovfs] = read_samples(round, flag, k): 
    cline = []; 
    label = []; 
    bovfs = []; 
    splitfile = import(splitfile); 
    for c = 1 : size(C,1): 
        for line in stipfile: 

        knn_train = knnsearch(mcentroids, x_train); 
        knn_test = knnsearch(mcentroids, x_test); 

        hist_train = build_bow(c_train, knn_train, nbins);
        hist_test = build_bow(c_test, knn_test, nbins);
        end; 
    end;
end;


function x_bow  = build_bow(cdata, xdata, nbins)
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

