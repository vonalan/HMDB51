function model = rbfnn()
    model = {};

    model.U = [];
    model.V = [];
    model.W = [];

    model.fit = @fit
    model.predict = @predict

    function rbfnn = fit(rbfnn, x_train, y_train)
        if isfield(param, 'centerCache')
            param.rngState = param.centerCache.rngState;
            U = param.centerCache.U;
        else
            param.rngState = rng; % save rng before kmeans
            [U, ~] = findCenter(Xtrain, Ytrain, param.nCenter);
        end;

        V = repmat(mean(pdist(U)), param.nCenter, 1);
        V = V * param.alpha;
        t1 = pdist2(Xtrain, U).^2; % ||x - u_j|| ^2
        t2 = exp(bsxfun(@rdivide, t1, (- 2 * (V.^2))'));
        W = lscov(t2, Ytrain);

        rbfnn.U = U;
        rbfnn.V = V;
        rbfnn.W = W;
    end;

    function out_testa = predict(rbfnn, x_testa, y_testa)
        t1 = pdist2(Xtest, rbfnn.U).^2;
        t2 = exp(bsxfun(@rdivide, t1, (- 2 * (rbfnn.V.^2))'));
        out_testa = t2 * rbfnn.W;
    end;

    function [ cn ] = centerNum(nCenter, nSample)
        assert(size(nSample, 1) == 1);
        assert(nCenter >= size(nSample, 2));
        assert(nCenter <= sum(nSample));

        cn = ones(size(nSample));
        nCenter = nCenter - sum(cn); %??? the reason why nCenter should be larger than number of class, but why?
        for i = 1:nCenter
           [~, j] = max(nSample ./ cn); % ??? why? rescale?
           cn(j) = cn(j) + 1;
        end;
    end;


    function [U, centerLabel] = findCenter(Xtrain, Ytrain, nCenter)
        if size(Ytrain, 2) == 1 % two-class case
            Ytrain = [Ytrain == 1, Ytrain == -1]; % concatenating horizontally and converting to 0-1 vector
        end

        Ytrain = (Ytrain == 1);
        assert(all(sum(Ytrain, 2) == ones(size(Xtrain, 1), 1))); % make sure no missing data

        cn = centerNum(nCenter, sum(Ytrain, 1)); % number of clusters of each class
        U = zeros(nCenter, size(Xtrain, 2));
        centerLabel = zeros(nCenter, 1);

        idx2 = 0;
        for i = 1:numel(cn) % ???
            idx1 = idx2 + 1;
            idx2 = idx2 + cn(i);
            centerLabel(idx1:idx2) = i;
            % clustering INSIDE each class
            % then put all centers together

            % for debug
            [~, centroids, sumd] = kmeans(Xtrain(Ytrain(:, i), :), cn(i), 'EmptyAction', 'singleton');

            [~, U(idx1:idx2, :)] = ... % codes are segment to two lines
                kmeans(Xtrain(Ytrain(:, i), :), cn(i), 'EmptyAction', 'singleton');
        end
        assert(idx2 == nCenter);
        % done
    end;

