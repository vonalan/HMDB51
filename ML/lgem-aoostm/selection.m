function [parent_pop, parent_objs] = selection(mixed_pop, mixed_objs, subproblems, params, idealpoint)
    % This process is a two-level one-one stable matching with incomplete  
    % lists. In general, the number of subproblems should be smaller than
    % that of the mixed_pop.  The purpuse of this process is to select a
    % near-optimal solution for each subproblem. And the matching between
    % subproblems and solutions is stable.
    global sol_related_size;
    
    Rmax = size(subproblems, 1);  % for new2
    
    num_objs   = size(mixed_objs, 2);
    size_objs   = size(mixed_objs, 1);
    size_subp   = size(subproblems, 1);
    sol_matrix_value  = zeros(size_objs, size_subp);
    subp_matrix_value = zeros(size_subp, size_objs);
    
    
    % Normalized the objective values of the mixed_pop into some limited range
    normalized_objs = normalization_nsga3(mixed_objs, size_objs, idealpoint);
    
    % Calculate the subproblem matrix and solution matrix
    for i = 1 : size_objs
        fitness = subobjective(subproblems, normalized_objs(i, :), zeros(1, num_objs), params.dmethod);
        sol_matrix_value(i, :) = distance(normalized_objs(i, :), subproblems);
        subp_matrix_value(:, i) = fitness;
    end
    
    % Sort the value matrix to get the preference rank matrix
    [~, sol_matrix]  = sort(sol_matrix_value, 2);
    [~, subp_matrix] = sort(subp_matrix_value, 2);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%% new2 %%%%%%%%%%%%%%%%%%%%%%%%
    sols = 1 : size_objs;
    subp_related_sol = zeros(size_subp, 1);                         %best sol in subp's sub region
    for i = 1 : size_subp
        related_sols = sols(sol_matrix(:, 1) == i);
        if ~isempty(related_sols)
            [~, min_index] = min(subp_matrix(i, related_sols));
            subp_related_sol(i) = related_sols(min_index);
        end
    end
    
    sol_related_size = ones(size_objs, 1);
    for i = 1 : size_objs
        for j = 2 : Rmax
            if subp_related_sol(sol_matrix(i, j)) ~= 0
                if all(mixed_objs(i, :) <= mixed_objs(subp_related_sol(sol_matrix(i, j)), :)) == true
                    break;
                end
            end
            sol_related_size(i) = j;
        end
    end
    sol_related_size(sol_related_size < num_objs) = num_objs;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    % matching with incomplete lists
    [matching_m, matching_w] = MGS_incomplete2(subp_matrix, sol_matrix, sol_related_size);

    
    % matching the remainings
    sol_matrix_value_rm = sol_matrix_value(matching_w == 0, matching_m == 0);
    subp_matrix_value_rm = subp_matrix_value(matching_m == 0, matching_w == 0);
    [~, sol_matrix_rm]  = sort(sol_matrix_value_rm, 2);
    [~, subp_matrix_rm] = sort(subp_matrix_value_rm, 2);
    matching_rm = MGS(subp_matrix_rm, sol_matrix_rm);
    sols = 1 : size_objs;
    unmantched_sols = sols(matching_w == 0);
    matching_m(matching_m == 0) = unmantched_sols(matching_rm);

    parent_pop  = mixed_pop(matching_m, :);
    parent_objs = mixed_objs(matching_m, :);
    
end