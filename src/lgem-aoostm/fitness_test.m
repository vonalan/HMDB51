% clear;
% clc;

global cline_train label_train stips_train cline_testa label_testa stips_testa centroids;
global R C K; % round, cates, K;


%*******************************
R = 1; 
K = 128;
ns1 = 5613856; 
ns2 = 2227730; 
nsr = 100000;
[C] = preload(R, K, ns1, ns2, nsr); 
%*******************************

%*******************************
mop = {}; 
mop.od = 2; 
mop.pd = K; 
popsize = 100; 

mop.domain = [zeros(1,mop.pd);ones(1,mop.pd)];

randarray  = rand(popsize, mop.pd);
lowend     = mop.domain(1, :);
span       = mop.domain(2, :) - lowend;
parent_pop = round(randarray .* (span(ones(1, popsize), :)) + lowend(ones(1, popsize), :));
%*******************************

O = fitness(parent_pop); 
fprintf('\n'); 