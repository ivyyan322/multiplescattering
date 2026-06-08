function params = fmm_params(N_four, box_size, k2, tol)
% FMM_PARAMS  Compute FMM parameters for single-level FMM.
%
%   P_tilde = 2P + k2*a*sqrt(2) + excess
%   Q = 2*P_tilde + 1  (matched quadrature)
%
%   Convergence requires P_tilde < k2 * 2 * a.
%   Feasibility requires a > 2P / (k2 * (2-sqrt(2))).

P = N_four;
digits = -log10(tol);
a = box_size;

% Base: 2P (harmonic pairs) + k2*a*sqrt(2) (worst-case offset bandwidth)
P_base = 2*P + k2*a*sqrt(2);

% Excess for accuracy: ~1 term per digit
P_excess = ceil(digits);

P_tilde = ceil(P_base + P_excess);

% Convergence limit
P_max = floor(k2 * 2 * a) - 1;

if P_tilde > P_max
    % Reduce excess
    P_tilde = min(P_tilde, P_max);
    if P_tilde < ceil(P_base)
        error(['FMM_PARAMS: Box too small. P_base=%.1f >= P_max=%d.\n' ...
               'Need box_size > %.2f.'], P_base, P_max, ...
               fmm_min_box_size(N_four, k2, tol));
    end
end

% Matched quadrature
Q = 2 * P_tilde + 1;
theta_q = (0:Q-1)' * (2*pi/Q);
kq = k2 * [cos(theta_q), sin(theta_q)];

params.P = P;
params.P_tilde = P_tilde;
params.Q = Q;
params.theta_q = theta_q;
params.kq = kq;
params.tol = tol;

end