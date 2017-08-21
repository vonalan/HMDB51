x_train_path = '../data/x_train.txt'; 
x_testa_path = '../data/x_testa.txt'; 
y_train_path = '../data/y_train.txt'; 
y_testa_path = '../data/y_testa.txt'; 

x_train = importdata(x_train_path);
x_testa = importdata(x_testa_path); 
y_train = importdata(y_train_path); 
y_testa = importdata(y_testa_path); 


for i = 1:10
    [param] = classifier_light([], y_train, x_train, [], y_testa, x_testa); 
    rbfnnC = param.rbfnnC; 
    
    acc_train = rbfnnC.acc_train;
    acc_testa = rbfnnC.acc_testa;
    err_train = mean(rbfnnC.err_train);
    err_testa = mean(rbfnnC.err_testa);
    stsm_train = mean(rbfnnC.stsm_train);
    lgem_train = mean(rbfnnC.lgem_train);
    
    fprintf('round: %d, acc_train: %.4f, acc_testa: %.4f, err_train: %.4f, err_testa: %.4f\n', ...
            i, acc_train, acc_testa, err_train, err_testa); 
end;