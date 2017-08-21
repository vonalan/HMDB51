function relative_objs = relativation(mixed_objs, popsize, idealpoint)

    min_objs = idealpoint(ones(popsize, 1), :);
    relative_objs = mixed_objs - min_objs;
end