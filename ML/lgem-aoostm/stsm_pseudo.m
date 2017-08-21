%%% SM computation;
%%% independent of classifier
%%% first used in MLPNN

%%% to calculate deltaX, generate 1000 feature points randomly around
%%% training samples with neighbor Q
function [stsm] = stsm_pseudo(x_train, W, U, V, Q, H) % deltaX is fluctuatio of trainX
    % H = 1000;
    [~,N] = size(x_train);
    deltaX = random('unif', -Q, Q, H, N); 
    
    tmpsum = 0;
    t1 = pdist2(x_train,U).^2;
    t2 = exp(bsxfun(@rdivide, t1, (-2 * (V .^ 2))'));
    f1 = t2 * W;
    for i = 1:H
        

        t1 = pdist2(bsxfun(@plus, x_train, deltaX(i,:)),U).^2;
        t2 = exp(bsxfun(@rdivide, t1, (-2 * (V .^ 2))'));
        f2 = t2 * W;
    
        tmpsum = tmpsum + (f2 - f1) .^ 2 ;
    end
    stsm = mean(tmpsum / H);
end