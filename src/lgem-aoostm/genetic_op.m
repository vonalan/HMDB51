function ind = genetic_op(curpoint_par, index, domain, params, matingindex)
    % This function uses 'DE/rand/1/bin' to generate a new solution
    %
    % subproblems : decomposition subproblems
    % index       : the index of the subproblem need to handle
    % domain      : the domain of the origional multiobjective problem
    % matingindex : the index of the neighborhood of a subproblem
    %
    % ind         : the generated individual
    
    
    %The random draw from the neighbours.
    nsize = length(matingindex);
    si    = ones(1, 3) * index;
    
    si(1) = matingindex(ceil(rand * nsize));
    while si(1) == index
        si(1) = matingindex(ceil(rand * nsize));
    end
    
    si(2) = matingindex(ceil(rand * nsize));
    while si(2) == index || si(2) == si(1)
        si(2) = matingindex(ceil(rand * nsize));
    end
    
%     si(3) = matingindex(ceil(rand * nsize));
%     while si(3) == index || si(3) == si(2) || si(3) == si(1)
%         si(3) = matingindex(ceil(rand * nsize));
%     end
    
    %retrieve the individuals.
    selectpoints = curpoint_par([index si], :);    

    oldpoint = curpoint_par(index, :);
    parDim   = size(domain, 2);
    
    jrandom = ceil(rand * parDim);
    
    %********************************************************************************************%
    randomarray         = rand(parDim, 1);
    deselect            = randomarray < params.CR;
    deselect(jrandom)   = true;
    newpoint            = selectpoints(1, :) + params.F * (selectpoints(2, :) - selectpoints(3, :));
    newpoint(~deselect) = oldpoint(~deselect);
    %********************************************************************************************%
    
    %repair the new value.
%     newpoint = max(newpoint, domain(1, :));
%     newpoint = min(newpoint, domain(2, :));
    
    for j = 1 : parDim
        %handle the boundary.
        lowbound = domain(1, j);
        upbound = domain(2, j);
        if (newpoint(j) < lowbound)
            %[r rnduni]=crandom(rnduni);
            r = rand;
            newpoint(j)= lowbound + r*(oldpoint(j)-lowbound);
        elseif (newpoint(j)>upbound)
            %[r rnduni]=crandom(rnduni);
            r = rand;
            newpoint(j)= upbound - r*(upbound - oldpoint(j));
        elseif (isnan(newpoint(j)))
            r = rand;
            newpoint(j)= r * (upbound - lowbound) + lowbound;
        end
    end
    
    % ********************************************************************%
    %ind = newpoint;
    ind = round(realmutate(newpoint, domain, 1 / parDim));
    %ind = round(gaussian_mutate(newpoint, 1 / parDim, domain));
    % ********************************************************************%
    
    clear si indexSet points selectpoints randomarray newpoint
end