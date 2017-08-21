 function parent_pop = randompoint(mop, popsize)
    % This function initializes the parent population
    %
    % mop        : The structure of the current MOP
    % popsize    : The population size of the paretn population
    %
    % parent_pop : The initialized parent population
    
    % be careful of round()
    %**********************************************************************
    randarray  = rand(popsize, mop.pd);
    lowend     = mop.domain(1, :);
    span       = mop.domain(2, :) - lowend;
    parent_pop = round(randarray .* (span(ones(1, popsize), :)) + lowend(ones(1, popsize), :));
    % parent_pop = randi([0,1], popsize, mop.pd); 
    %**********************************************************************
end
