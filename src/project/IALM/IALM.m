function [A_hat, E_hat, Y, mu, iter] = IALM(D, lambda, tol, maxIter)

[m,n] = size(D);

if(nargin < 2) lambda = 1 / sqrt(m); end
if(nargin < 3) tol = 1e-7; elseif(tol == -1) tol = 1e-7; end
if(nargin < 4) maxIter = 1000; elseif(maxIter == -1) maxIter = 1000; end

% initialize
Y = D;
norm_two = norm(Y, 2);
norm_inf = norm(Y(:), inf) / lambda;
dual_norm = max(norm_two, norm_inf);
Y = Y / dual_norm;

A_hat = zeros(m, n);
E_hat = zeros(m, n);
mu = 1.25 / norm_two;
mu_bar = mu * 1e7;
rho = 1.5;
d_norm = norm(D, 'fro');

iter = 0;
num_svd = 0;
converged = false;
stop_condition = 1;
sv = 10;
while ~converged       
    iter = iter + 1;
    temp_T = D - A_hat + (1 / mu) * Y;
    E_hat = max(temp_T - lambda / mu, 0);
    E_hat = E_hat + min(temp_T + lambda / mu, 0);
    
    % a fast SVD
    [U,S,V] = svd(D - E_hat + (1 / mu) * Y, 'econ');
    diagS = diag(S);
    svp = length(find(diagS > 1/mu));
    
    if svp < sv
        sv = min(svp + 1, n);
    else
        sv = min(svp + round(0.05 * n), n);
    end
    
    A_hat = U(:, 1:svp) * diag(diagS(1:svp) - 1 / mu) * V(:, 1:svp)';    
    num_svd = num_svd + 1;
    Z = D - A_hat - E_hat;
    Y = Y + mu * Z;
    mu = min(mu * rho, mu_bar);

    %% stop Criterion    
    stop_condition = norm(Z, 'fro') / d_norm;
    
    if stop_condition < tol
        converged = true;
    end    
    
    if ~converged && iter >= maxIter
        converged = true ;       
    end
end