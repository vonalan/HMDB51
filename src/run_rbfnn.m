function [ mmse, mstsm, trainacc, testacc ] = run_rbfnn(hist_train, hist_test, y_train, y_test)

param = {};
param.nRound = 1;
param.nCenterList = (255:255)';
param.alphaList =  logspace(0, 0, 1)';
% param.testRatio = 0.5;
param.Q = logspace(-1, -1, 1)';

nFeature = size(hist_train, 2);
nRound = param.nRound;
nCenterList = param.nCenterList;
alphaList = param.alphaList;
Q = param.Q;

% matlab cell is column major ordered
%                     k,               j,              i
result = cell(numel(alphaList), numel(nCenterList), nRound);
for i = 1:nRound
    for j = 1:numel(nCenterList)
        centerCache = struct();
        for k = 1:numel(alphaList)
            caseParam = {};

            caseParam.nCenter = nCenterList(j);
            caseParam.alpha = alphaList(k);
            
            % preprocessing
            [hist_train, hist_test, ~, ~] = preProcesse(hist_train, hist_test);
            
            % train
            if k > 1
                caseParam.centerCache = centerCache;
            end
            caseParam = trainRBFNN(hist_train, y_train, caseParam);
            % save cache
            centerCache.U = caseParam.U;
            centerCache.rngState = caseParam.rngState;
            
            % test
            outputTrain = testRBFNN(hist_train, caseParam);
            outputTest = testRBFNN(hist_test, caseParam);
            
            % testAcc
            [~,oidx1] = max(outputTest,[],2); 
            [~,lidx1] = max(Ytest,[],2);
            idx1 = oidx1 == lidx1;
            testAcc = sum(idx1)/numel(idx1); 
            caseParam.testAcc = testAcc;
            
            % trainAcc
            [~,oidx2] = max(outputTrain,[],2); 
            [~,lidx2] = max(Ytrain,[],2);
            idx2 = oidx2 == lidx2;
            trainAcc = sum(idx2)/numel(idx2); 
            caseParam.trainAcc = trainAcc;
            
            % Remp, mean squared error, please note that the error is SQUARED
            RempTrain = mean((outputTrain - y_train).^2, 1);
            RempTest = mean((outputTest - y_test).^2, 1);
            assert(2 == same(size(RempTrain), size(RempTest)));
            assert(2 == same(size(RempTrain), [1, size(y_train, 2)]));
            caseParam.RempTrain = RempTrain;
            caseParam.RempTest = RempTest;
            
            % STSM
            caseParam = stsm_pseudo(hist_train, caseParam.W, caseParam.U, caseParam.V, Q);
            % part of formular (7)
            % RsmScore = sqrt(Remp) + sqrt(ESQ)
            % since error is computed in Euclidean function
            % in a multi-class problem,
            % squared error on every output dimension are summarize
            assert(size(RempTrain, 2) == size(caseParam.STSM, 2));
            % 
%             caseParam.RsmScore = sqrt(sum(RempTrain, 2)) + sqrt(sum(caseParam.STSM, 2));
%             assert(2 == same(size(caseParam.RsmScore), size(Q)));
            %
%             caseParam.trainMask = trainMask(:, i);
%             caseParam.testMask = testMask(:, i);
            % done
            result{k, j, i} = caseParam;
            
            mmse = mean(caseParam.RempTrain); 
            mstsm = mean(caseParam.STSM); 
            trainacc = caseParam.trainAcc; 
            testacc = caseParam.testAcc; 
            
            fprintf('numcenter: %d, trainAcc: %.8f, testAcc: %.8f, mse: %.8f, stsm: %.8f. \n', nFeature, trainAcc, testAcc, mmse, mstsm); 
            
        end
    end
end

end