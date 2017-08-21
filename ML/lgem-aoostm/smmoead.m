function [pareto, parent_pop, results] = smmoead(mop, varargin)

    % Global variable definition.
    global params idealpoint objDim parDim itrCounter num_eval oldobj utility;
    
    % Initialization and set the algorithms parameters.
    paramIn = varargin;
    [objDim, parDim, idealpoint, params, parent_pop, parent_objs, subproblems, mating_neighbour] = init(mop, paramIn);
    
    utility = ones(params.popsize, 1);
    oldobj = subobjective(subproblems, parent_objs, idealpoint, params.dmethod);
    
    itrCounter = 1;
    num_eval = params.popsize;
    
    while ~terminate(num_eval)
        
        % Reproduction
        [offspring, offspring_objs] = evolve(parent_pop, mop, mating_neighbour, params);
        
        % Merge
        mixed_pop  = [parent_pop; offspring];
        mixed_objs = [parent_objs; offspring_objs];
        
        % Update the ideal point
        idealpoint = min(idealpoint, min(mixed_objs));
        
        % Enviromental Selection (Stable Matching)
        [parent_pop, parent_objs] = selection(mixed_pop, mixed_objs, subproblems, params, idealpoint);
       
        % fprintf('iteration %d finished...\n', itrCounter);
        fprintf('evaluation %d finished...\n', num_eval);
        
        itrCounter = itrCounter + 1;      
        if (rem(itrCounter,30) == 0) % updating of the utility.
            util_update(parent_objs, subproblems); 
            
        end  
        if mop.od == 2
            scatter(parent_objs(:, 1), parent_objs(:, 2)); drawnow
        else
            scatter3(parent_objs(:, 1), parent_objs(:, 2), parent_objs(:, 3)); drawnow
        end
    end
    
    [pareto, results] = mop.func(parent_pop);
% %    figure
%     if mop.od == 2
%         plot(pareto(:, 1), pareto(:, 2), 'o');
%     else
%         plot3(pareto(:, 1), pareto(:, 2), pareto(:, 3), 'o');
%     end
    
end

function y = terminate(num_eval)

    global params;
    % y = itrcounter > params.iteration;
    y = num_eval >= params.evaluation;
    
end