function distance = distance(ind, weight)
    % This function is used to calculate the perpendicular distances
    % between the 'ind' and reference lines related to the weight vectors

    weight_size = size(weight, 1);

    ind_matrix  = ind(ones(1, weight_size), :);
    temp1       = sum(weight .* ind_matrix, 2) ./ sum(weight .* weight, 2);
    temp1       = temp1(:, ones(1, size(weight, 2)));
    proj_matrix = temp1 .* weight;
    orthogonal  = ind_matrix - proj_matrix;
    
%     distance    = sqrt(sum(orthogonal .* orthogonal, 2));
    distance    = sum(orthogonal .* orthogonal, 2);
    distance    = distance';

end

