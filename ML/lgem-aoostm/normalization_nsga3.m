function normalized_objs = normalization_nsga3(mixed_objs, popsize, idealpoint)

    global nadirpoint
    
    num_objs        = size(mixed_objs, 2);
    min_objs        = idealpoint(ones(popsize, 1), :);
    
    extreme_points = zeros(num_objs, num_objs);
    for i = 1 : num_objs 
        weight = zeros(1, num_objs);
        weight(i) = 1;
        
        subobjs = subobjective(weight, mixed_objs, idealpoint, 'i_te');
        [~, min_index] = min(subobjs);
        extreme_points(i, :) = mixed_objs(min_index, :);
    end
    
    degenerate = false;
    A = extreme_points - idealpoint(ones(num_objs, 1), :);
    b = ones(1, num_objs);
    for p = 1 : num_objs
        maxValue = p;

        for i = p + 1 : num_objs
            if abs(A(i, p)) > abs(A(maxValue, p))
                maxValue = i;
            end
        end

        temp = A(p, :);
        A(p, :) = A(maxValue, :);
        A(maxValue, :) = temp;

		t = b(p);
        b(p) = b(maxValue);
        b(maxValue) = t;

        if abs(A(p, p)) <= 10 ^ (-10)
            degenerate = true;
            break;
        end

        for i = p + 1 : num_objs
            alpha = A(i, p) / A(p, p);
            b(i) = b(i) - alpha * b(p);

            for j = p : num_objs
                A(i, j) = A(i, j) - alpha * A(p, j);
            end
        end
    end
    
    if ~degenerate
		x = zeros(1, num_objs);
        
        for i = 1 : num_objs
            sum_value = 0;
            for j = num_objs - i + 2 : num_objs
                sum_value = sum_value + A(num_objs - i + 1, j) * x(j);
            end
            
            x(num_objs - i + 1) = (b(num_objs - i + 1) - sum_value) / A(num_objs - i + 1, num_objs - i + 1);
        end

        intercepts = 1 ./ x;
        
        if sum(intercepts < 0.001) ~= 0
            degenerate = true;
        end
    end
    
    
%     max_obj         = max(extreme_points);

    if degenerate
        nadirpoint      = max(mixed_objs);
    else
        nadirpoint        = intercepts + idealpoint;
    end
    
%     nadirpoint=[2 4 6];
    max_objs        = nadirpoint(ones(popsize, 1), :);
    
    normalized_objs = (mixed_objs - min_objs) ./ (max_objs -  min_objs);
end