function [matching_m, matching_w] = MGS_incomplete2(man_matrix, woman_matrix, woman_refer_size)

size_man     = size(man_matrix, 1);
size_woman   = size(woman_matrix, 1);

reject_times = zeros(1, size_man);
matching_m   = zeros(1, size_man);
matching_w   = zeros(1, size_woman);


while ~isequal(matching_m == 0 & reject_times ~= size_woman, zeros(1, size_man))
    men = find(matching_m == 0 & reject_times ~= size_woman);
    man_i = men(1);
    woman_i = man_matrix(man_i, reject_times(man_i) + 1);
    
    m_i_rank = find(woman_matrix(woman_i, :) == man_i, 1);
    if m_i_rank > woman_refer_size(woman_i)
        reject_times(man_i) = reject_times(man_i) + 1;
    else
        if matching_w(woman_i) == 0
            matching_w(woman_i) = man_i;
            matching_m(man_i) = woman_i;
        else
            if m_i_rank < find(woman_matrix(woman_i, :) == matching_w(woman_i), 1)
                matching_m(matching_w(woman_i)) = 0;
                reject_times(matching_w(woman_i)) = reject_times(matching_w(woman_i)) + 1;
                matching_w(woman_i) = man_i;
                matching_m(man_i) = woman_i;
            else
                reject_times(man_i) = reject_times(man_i) + 1;
            end
        end
    end
    
end


