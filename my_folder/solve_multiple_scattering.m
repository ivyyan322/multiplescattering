function [beta_vec, flag, relres, iter] = solve_multiple_scattering(X_proto, ...
    centers, rotations, k2, N_four, a_vec, method, tol, maxit)
% SOLVE_MULTIPLE_SCATTERING  Solve the multiple scattering system.
%
%   [beta_vec, flag, relres, iter] = solve_multiple_scattering(X_proto, ...
%       centers, rotations, k2, N_four, a_vec, method, tol, maxit)
%
%   Solves (I - X*T) * beta = X * alpha for the outgoing coefficients beta.
%
%   Inputs:
%     X_proto   - (Nmodes x Nmodes) scattering matrix for prototype shape
%     centers   - M x 2 array of particle center coordinates
%     rotations - M x 1 vector of rotation angles (radians)
%     k2        - exterior wavenumber
%     N_four    - cylindrical harmonics truncation
%     a_vec     - (Nmodes*M x 1) incident field coefficients
%     method    - 'gmres' or 'direct' (default: 'gmres')
%     tol       - GMRES tolerance (default: 1e-10)
%     maxit     - GMRES max iterations (default: 200)
%
%   Outputs:
%     beta_vec  - (Nmodes*M x 1) outgoing coefficients
%     flag      - GMRES convergence flag (0 = converged)
%     relres    - relative residual
%     iter      - iteration count

if nargin < 7 || isempty(method); method = 'gmres'; end
if nargin < 8 || isempty(tol);    tol = 1e-10;      end
if nargin < 9 || isempty(maxit);  maxit = 200;       end

M = size(centers, 1);
Nmodes = 2*N_four + 1;

%% Build block-diagonal scattering matrix
X_cells = cell(M, 1);
for m = 1:M
    X_cells{m} = rotate_scattering_matrix(X_proto, rotations(m), N_four);
end
X_blk = blkdiag(X_cells{:});

%% Build translation matrix
T_full = build_translation_matrix(centers, k2, N_four);

%% Right-hand side
rhs = X_blk * a_vec;

%% Solve
sys_mat = eye(Nmodes*M) - X_blk * T_full;

if strcmpi(method, 'direct')
    beta_vec = sys_mat \ rhs;
    flag = 0;
    relres = norm(sys_mat*beta_vec - rhs) / norm(rhs);
    iter = 0;
else
    % GMRES (no restart for now; restart = Nmodes*M)
    restart = min(Nmodes*M, 100);
    [beta_vec, flag, relres, iter] = gmres(sys_mat, rhs, restart, tol, maxit);
end

end
