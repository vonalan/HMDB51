function mop = testmop(testname)
    %Get test multi-objective problems from a given name.
    %   The method get testing or benchmark problems for Multi-Objective
    %   Optimization. The implemented problems included ZDT, OKA, KNO.
    %   User get the corresponding test problem, which is an instance of class
    %   mop, by passing the problem name and optional dimension parameters.
    
    global K; 
    
    mop = struct('name', [], 'od', [], 'pd', [], 'domain', [], 'func', []);

    switch lower(testname)
        case 'mse-stsm'
            mop = BIO(mop, testname);
        case 'mse'
            mop = SIG(mop, testname);
    end; 

    % ********************************************************************%
    % BIO function generator
    function p = BIO(p, testname)
        p.name   = testname; 
        p.od     = 2;
        p.pd     = K;
        % p.domain = [zeros(1, dim); ones(1, dim)];
        p.domain = [zeros(1, p.pd); ones(1, p.pd)];
        p.func   = @evaluate;

        % ********************************************************************%
        % redefining objective function %
        function [Y, results] = evaluate(X)
            % long term objective
            % x = [k1, m1, k2, m2, k3, m3]
            % y = [mse3, stsm3, mse3, stsm3, mse3, stsm3]

            % short term objective
            % x = [k, m]
            % y = [mse, stsm]

            results = fitness(X); 
            Y = results(:,1:p.od);
        end
        % ********************************************************************%
    end
    % ********************************************************************%

    % SIG function generator
    function p = SIG(p, testname)
        p.name   = testname; 
        p.od     = 2;
        p.pd     = K;
        % p.domain = [zeros(1, dim); ones(1, dim)];
        p.domain = [zeros(1, p.pd); ones(1, p.pd)];
        p.func   = @evaluate;

        % ********************************************************************%
        % redefining objective function %
        function [Y, results] = evaluate(X)
            % long term objective
            % x = [k1, m1, k2, m2, k3, m3]
            % y = [mse3, stsm3, mse3, stsm3, mse3, stsm3]

            % short term objective
            % x = [k, m]
            % y = [mse, stsm]

            results = fitness(X); 

            Y = zeros(size(results,1),2); 
            Y(:,1) = results(:,1);
        end
        % ********************************************************************%
    end
    % ********************************************************************%
end