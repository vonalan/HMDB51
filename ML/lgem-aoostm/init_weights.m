function [weight, neighbour] = init_weights(popsize, niche, objDim)
    % init_weights function initialize a pupulation of subproblems structure
    % with the generated decomposition weight and the neighbourhood
    % relationship.
    
    if objDim == 8 
        weight1 = initweight(objDim, 120);
        weight2 = initweight(objDim, 36);
        weight2 = 0.5 / objDim + 0.5 .* weight2;
        weight = [weight1 weight2];
    elseif objDim == 10 
        weight1 = initweight(objDim, 220);
        weight2 = initweight(objDim, 55);
        weight2 = 0.5 / objDim + 0.5 .* weight2;
        weight = [weight1 weight2];     
    elseif objDim == 1
        weight = initweight(objDim, popsize);
    else 
        weight = initweight(objDim, popsize);
    end
    weight = weight';
    %weight = normr(weight);

    %Set up the neighbourhood.
    leng           = size(weight, 1);
    distanceMatrix = zeros(leng, leng);
    neighbour      = zeros(leng, niche);
    
    for i = 1 : leng
        for j = i + 1 : leng
            A                    = weight(i, :);
            B                    = weight(j, :);
            distanceMatrix(i, j) = (A - B) * (A - B)';
            distanceMatrix(j, i) = distanceMatrix(i, j);
        end
        [~, sindex]     = sort(distanceMatrix(i, :));
        neighbour(i, :) = sindex(1 : niche)';
    end
end