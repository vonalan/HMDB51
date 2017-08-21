%**************************************************************************************************
% Author: Mengyuan Wu :: http://coda-group.github.io/mwu.html
% Reference: M. Wu, K. Li, S. Kwong, Y. Zhou, Q. Zhang, Matching-based selection with incomplete lists for decomposition multi-objective optimization, IEEE Transactions on Evolutionary Computation, 2017.
% Copyright 2016: City University of Hong Kong.
%**************************************************************************************************

clc;
clear;
tic;

format long;
format compact;

global cline_train label_train stips_train cline_testa label_testa stips_testa centroids;
global R C K; % round, cates, K;

%*******************************
R = 1; 
K = 4096;
ns1 = 5613856; 
ns2 = 2227730; 
nsr = 100000;
[C] = preload(R, K, ns1, ns2, nsr); 
%*******************************

% rand('seed',0)
rand('state', sum(100*clock));

problems = {'mse', 'mse-stsm'}; 

% length = length(problems);
totalrun       = 1;

i = 1;
problem = problems{i};
fprintf('Running on %s...\n', problem);
for j = 1 : totalrun

    mop    = testmop(problem);
    [pareto, ps, metrics] = smmoead(mop);
    
    % eval(['save ' sprintf(problems{i}) ' pareto']);
    path = ['../data/results', '_', problem, ''];
    save(path, 'pareto', 'ps', 'metrics');
end;

% toc;
% exit;