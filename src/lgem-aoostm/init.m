function [objDim, parDim, idealp, params, pop, objs, subproblems, neighbour] = init(mop, propertyArgIn)
    % Set up the initial setting for the MOEA/D.    
    objDim = mop.od;
    parDim = mop.pd;    
    idealp = ones(1, objDim) * inf;
    
    % Default values for the parameters.
    
    %*********************************************************************%
    params.CR        = 0.5;
    params.popsize    = 100;
    params.evaluation = 1000;
    %*********************************************************************%
    
    params.dmethod   = 'i_te';
    
    params.F         = 0.5;
    
    %*********************************************************************%
    params.nr        = 2;
    params.niche     = 20;
    params.delta     = 0.9;
    %*********************************************************************%
    
    params.dynamic   = 1;
    params.selportion = 5;
    
    % Handle the parameters, mainly about the popsize
    while length(propertyArgIn) >= 2
        prop          = propertyArgIn{1};
        val           = propertyArgIn{2};
        propertyArgIn = propertyArgIn(3 : end);

        switch prop
            case 'popsize'
                params.popsize = val;
            case 'niche'
                params.niche = val;
            case 'iteration'
                params.iteration = val;
            case 'method'
                params.dmethod = val;
            otherwise
                warning('MOEA does not support the given parameters name!');
        end
    end
    % Initialize the direction vectors (subproblems)
    [subproblems, neighbour] = init_weights(params.popsize, params.niche, objDim);
    params.popsize           = length(subproblems);
    
    % Initialize the population
    pop  = randompoint(mop, params.popsize);
    [objs, results] = mop.func(pop);
    fprintf('the original population is generated! ');
    
    % Find the ideal point
    idealp = min(idealp, min(objs));
    
end