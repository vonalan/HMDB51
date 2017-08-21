function util_update(new_pop, subproblems)
%Search Utility updation.
%   This function update the subproblem's search utility, using the
%   improvement of the subproblems' objective vaue since the last caculation.
    
    global idealpoint params oldobj utility;
    shrehold = 0.001;

    newobj = subobjective(subproblems, new_pop, idealpoint, params.dmethod);
    
    delta = (oldobj - newobj) ./ oldobj;

    utility(delta >= shrehold) = deal(1);   
    utility(delta <= 0) = utility(delta <= 0) .* 0.95; 
    update = delta < shrehold & delta > 0;
    utility(update) = utility(update) .* (delta(update) ./ shrehold .* 0.05 +0.95);   
    oldobj = newobj;
end
