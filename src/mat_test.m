function [ mmse, mstsm, trainacc, testacc ] = select_model(c_train, y_train, x_train, c_testa, y_testa, x_testa)
    param = {};
    rbfnn = run_rbfnn(c_train, y_train, x_train, c_testa, y_testa, x_testa)
end;


function [rbfnn] = run_rbfnn(c_train, y_train, x_train, c_testa, y_testa, x_testa)
    [x_train, bias, scale] = auto_scale(x_train, 0, 1, 'train')
    [x_testa, _, _] = auto_scale(x_testa, bias, scale, 'testa')

    rbfnnC = rbfnn();
    rbfnnC = fit(rbfnnC, x_train, y_train)
    out_train = predict(rbfnnC, x_train)
    out_testa = predict(rbfnnC, x_testa)

    acc_train = calc_acc(y_train, out_train);
    acc_testa = calc_acc(y_train, out_train);

    err_train = calc_err(y_testa, out_testa);
    err_testa = calc_err(y_testa, out_testa);

    stsm_train = stsm_pseudo(x_train, rbfnnC.W, rbfnnC.U, rbfnnC.V, Q);
    lgem_train = sqrt(sum(err_train, 2)) + sqrt(sum(stsm, 2));
    % stsm_testa = stsm_pseudo(x_testa, rbfnnC.W, rbfnnC.U, rbfnnC.V, Q);
    % lgem_testa = sqrt(sum(err_testa, 2)) + sqrt(sum(stsm_testa, 2));

    rbfnn.acc_train = acc_train;
    rbfnn.acc_testa = acc_testa;
    rbfnn.err_train = err_train;
    rbfnn.err_testa = err_testa;
    rbfnn.stsm_train = stsm_train;
    rbfnn.stsm_testa = stsm_testa;
    rbfnn.lgem_train = lgem_train;
    rbfnn.lgem_testa = lgem_testa;
end;


function [dataset, bias, scale] = auto_scale(dataset, xbias, xscale, mode)
    bias = xbias;
    scale = xscale;

    if mode == 'train':
        xmin = dataset.min()
        xmax = dataset.max()
        bias = (xmax + xmin)/2.0
        scale = xmax - bias
    end;

    dataset = bsxfun(@rdivide, bsxfun(@minus, dataset, bias), scale);
end;


function err = calc_err(y_true, y_predict))
    err = mean((y_predict - y_true).^2, 1);
end;


function acc = calc_acc(y_true, y_predict))
    [~,pidx] = max(y_predict,[],2);
    [~,tidx] = max(y_true,[],2);
    idx = pidx == tidx;
    trainAcc = sum(idx)/numel(idx);
end;


