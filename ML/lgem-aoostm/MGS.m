function [matching, man_reduced_matrix, woman_reduced_matrix] = MGS(man_matrix, woman_matrix)

size_man = size(man_matrix, 1);
accept_flag      = zeros(1, size_man);
reject_times     = ones(1, size_man);

count            = 0;
fixed_index      = 1 : size_man;

while count ~= size_man

    matching            = man_matrix(sub2ind(size(man_matrix), fixed_index, reject_times));
    process_array       = zeros(1, size_man);
    process_array_index = 1;

    for i = 1 : size_man

        if sum(process_array == i) ~= 0
            continue; 
        else
            [~, temp_index] = find(matching == matching(i));
            if size(temp_index) == 1
                process_array(process_array_index) = temp_index;
                process_array_index                = process_array_index + 1;
                accept_flag(temp_index)            = 1;
            else
                process_array(process_array_index : process_array_index + length(temp_index) - 1)  = temp_index;
                process_array_index                      = process_array_index + length(temp_index);
                subp_current                             = woman_matrix(matching(i), :);
                [~, subp_index, ~]                       = intersect(subp_current, temp_index);
                optimal_index                            = min(subp_index);
                accept_flag(temp_index)                  = 0;
                accept_flag(subp_current(optimal_index)) = 1;
            end
        end
    end

    count        = sum(accept_flag == 1);
    reject_times = reject_times + (accept_flag == 0);

end

matching = man_matrix(sub2ind(size(man_matrix), fixed_index, reject_times));