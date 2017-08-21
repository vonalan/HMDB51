function [param] = classifier_light(c_train, y_train, x_train, c_testa, y_testa, x_testa)
    param = {}; 

    param.round = 1; 
    param.num_center = 255;
    param.alpha = 1.0; 
    param.split_ratio = 0.5; 
    param.Q = 0.1; 
    param.K = 128; 
    param.H = 1000; 

    [rbfnnC] = run_rbfnn(param, c_train, y_train, x_train, c_testa, y_testa, x_testa); 
    param.rbfnnC = rbfnnC; 
end


function [rbfnnC] = run_rbfnn(param, c_train, y_train, x_train, c_testa, y_testa, x_testa)    
    [x_train, bias, scale] = auto_scale(x_train, 0, 1, 'train');
    [x_testa, ~, ~] = auto_scale(x_testa, bias, scale, 'valid');

    rbfnnC = rbfnn();
    rbfnnC = rbfnnC.fit(param, rbfnnC, x_train, y_train);
    out_train = rbfnnC.predict(rbfnnC, x_train);
    out_testa = rbfnnC.predict(rbfnnC, x_testa);

    acc_train = calc_acc(y_train, out_train);
    acc_testa = calc_acc(y_testa, out_testa);

    err_train = calc_err(y_train, out_train);
    err_testa = calc_err(y_testa, out_testa);

    stsm_train = stsm_pseudo(x_train, rbfnnC.W, rbfnnC.U, rbfnnC.V, param.Q, param.H);
    lgem_train = (sqrt(err_train) + sqrt(stsm_train)).^2;
    % stsm_testa = stsm_pseudo(x_testa, rbfnnC.W, rbfnnC.U, rbfnnC.V, case_param.Q, case_param.H);
    % lgem_testa = sqrt(sum(err_testa, 2)) + sqrt(sum(stsm_testa, 2));

    rbfnnC.acc_train = acc_train;
    rbfnnC.acc_testa = acc_testa;
    
    rbfnnC.err_train = err_train;
    rbfnnC.err_testa = err_testa;
    
    rbfnnC.stsm_train = stsm_train;
    rbfnnC.lgem_train = lgem_train;
    
    % rbfnnC.stsm_testa = stsm_testa;
    % rbfnnC.lgem_testa = lgem_testa;
end


function [dataset, bias, scale] = auto_scale(dataset, xbias, xscale, mode)
    bias = xbias;
    scale = xscale;

    if strcmp(mode, 'train')
        xmin = min(dataset); 
        xmax = max(dataset); 
        bias = (xmax + xmin)/2.0;
        scale = xmax - bias;
    end;

    dataset = bsxfun(@rdivide, bsxfun(@minus, dataset, bias), scale);
end


function [err] = calc_err(y_true, y_predict)
    err = mean((y_predict - y_true).^2, 1);
end


function [acc] = calc_acc(y_true, y_predict)
    [~,pidx] = max(y_predict,[],2);
    [~,tidx] = max(y_true,[],2);
    idx = pidx == tidx;
    acc = sum(idx)/numel(idx);
end


