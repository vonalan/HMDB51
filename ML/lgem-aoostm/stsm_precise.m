function [ param ] = stsm_precise(param, Xtrain, Q)
% Stochastic Sensitivity Measure
% formula (9)
% compute param for multiple Q, Q should be a column vector
assert(size(Q, 2) == 1);

% central moments of every feature
mu_xi = mean(Xtrain, 1);
sigma2_xi = var(Xtrain, 1); % this variance is normalized with N, not N-1
E3 = moment(Xtrain, 3);
E4 = moment(Xtrain, 4);

% Var(s_j)
assert(size(param.U, 1) == param.nCenter);
t1 = bsxfun(@minus, mu_xi, param.U); % mu_xi - U_ji
t3 = bsxfun(@times, 4 * sigma2_xi, t1.^2); % 3rd term
t4 = bsxfun(@times, 4 * E3, t1); % 4th term

Var_s = sum(E4, 2) - sum(sigma2_xi.^2, 2) + sum(t3, 2) + sum(t4, 2);
assert(same(size(Var_s), [param.nCenter, 1]) == 2);
% E(s_j)
E_s = sum(sigma2_xi, 2) + sum(t1.^2, 2);
assert(same(size(Var_s), [param.nCenter, 1]) == 2);
assert(same(size(E_s), [param.nCenter, 1]) == 2);

t5 = Var_s./(2*(param.V.^4)) - E_s./(param.V.^2);
log_phi = bsxfun(@plus, log(abs(param.W)) * 2, t5);
%phi = bsxfun(@times, param.W.^2, exp(Var_s./(2*(param.V.^4)) - E_s./(param.V.^2)));
% log(phi) == logPhi

assert(all(param.V(:) > 0));
assert(same(size(param.V), [param.nCenter, 1]) == 2);

log_upsilon = bsxfun(@plus, log_phi, log(sum(sigma2_xi, 2) + sum(t1.^2, 2)) - 4 * log(param.V));
% upsilon = bsxfun(@times, phi, sum(bsxfun(@rdivide, bsxfun(@plus, sigma2_xi, t1.^2), param.V.^4), 2));
% log(upsilon) == log_upsilon

log_zeta = bsxfun(@minus, log_phi, 4 * log(param.V));
% zeta = bsxfun(@rdivide, phi, param.V.^4);

sum_upsilon = sum(exp(log_upsilon), 1);
sum_zeta = sum(exp(log_zeta), 1);

result = 1/3 .* bsxfun(@times, Q.^2, sum_upsilon) + ...
    0.2/9 .* size(Xtrain, 2) .* bsxfun(@times, Q.^4, sum_zeta);

assert(all(size(result) == [size(Q, 1), size(param.W, 2)]));
assert(all(size(log_upsilon) == [param.nCenter, size(param.W, 2)]));
assert(all(size(log_zeta) == [param.nCenter, size(param.W, 2)]));

param.STSM = result;
% param.log_upsilon = log_upsilon;
% param.log_zeta = log_zeta;
% param.Q = Q;

end

