function [offspring, objs] = evolve(parent_pop, mop, mating_neighbour, params)
    % This function used to generate the offspring population
    %
    % parent_pop -> the parent population
    % mop        -> the structure of the MOP
    % params     -> the structure of the parameters
    %
    % offspring  -> the offspring population
    % objs       -> the objective values of the offspring population

    
    % select the subproblem according to its utility.
    % if params.dynamic is not true, then no selection is used.
    global num_eval;
    
    % ********************************************************************%
    if (params.dynamic)
%         if (rem(itrCounter,30) == 0) 
%             selindex = 1 : params.popsize; 
%         else
            selindex = util_select();
%         end  
    else
        selindex = 1 : params.popsize;  
    end
    % ********************************************************************%
    
    % objs = zeros(length(selindex), mop.od);
    offspring = zeros(length(selindex), mop.pd);
    
    % ********************************************************************%
    % for parfor
    % temp_index = zeros(length(selindex), 1); 
    % temp_offspring = zeros(length(selindex), mop.pd);
    % temp_objs = zeros(length(selindex), mop.od);
    
    for i = 1 : length(selindex)
        % bug bug bug %
        % temp_index = [temp_index; [i]]; 
        
        index = selindex(i);
        if (rand < params.delta)
            matingindex = mating_neighbour(index, :);
        else
            matingindex = (1 : params.popsize);
        end
        
        offspring(i, :) = genetic_op(parent_pop, index, mop.domain, params, matingindex);
        % objs(i, :)      = mop.func(offspring(i, :));
        % [o, res] = mop.func(offspring(i, :));
        % objs(i, :) = o; 
        
        num_eval = num_eval + 1;
    end
    
    % temp_offspring = offspring(temp_index,:); 
    % temp_objs = mop.func(temp_offspring);
    % objs(temp_index,:) = temp_objs; 
    [objs, ~] = mop.func(offspring); 
    % ********************************************************************%
    
end