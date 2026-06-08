function [beta_vec, flag, relres, iter, t_info] = solve_multiple_scattering_fmm(...
    X_proto, centers, rotations, k2, N_four, a_vec, tol_fmm, tol_gmres, maxit)
% SOLVE_MULTIPLE_SCATTERING_FMM  Solve multiple scattering with FMM-accelerated GMRES.
%
%   [beta_vec, flag, relres, iter, t_info] = solve_multiple_scattering_fmm(...)
%
%   Solves (I - X*T) * beta = X * alpha using GMRES, where the T*beta
%   matrix-vector product is computed via FMM instead of dense matrix.
%
%   Inputs:
%     X_proto   - (Nmodes x Nmodes) scattering matrix for prototype shape
%     centers   - M x 2 particle centers
%     rotations - M x 1 rotation angles (radians)
%     k2        - exterior wavenumber
%     N_four    - cylindrical harmonics truncation
%     a_vec     - (Nmodes*M x 1) incident field coefficients
%     tol_fmm   - FMM accuracy tolerance (default: 1e-6)
%     tol_gmres - GMRES convergence tolerance (default: 1e-6)
%     maxit     - GMRES max iterations (default: 200)
%
%   Outputs:
%     beta_vec  - (Nmodes*M x 1) outgoing coefficients
%     flag      - GMRES convergence flag (0 = converged)
%     relres    - relative residual
%     iter      - iteration count
%     t_info    - struct with timing info for each stage

if nargin < 7 || isempty(tol_fmm);   tol_fmm = 1e-6;  end
if nargin < 8 || isempty(tol_gmres);  tol_gmres = 1e-6; end
if nargin < 9 || isempty(maxit);      maxit = 200;       end

M = size(centers, 1);
Nmodes = 2*N_four + 1;
Ndof = Nmodes * M;

%% Stage 1: Build block-diagonal scattering matrix X
tic;
X_cells = cell(M, 1);
for m = 1:M
    X_cells{m} = rotate_scattering_matrix(X_proto, rotations(m), N_four);
end
X_blk = blkdiag(X_cells{:});
t_info.t_Xblk = toc;
fprintf('  X_blk assembly: %.2f s\n', t_info.t_Xblk);

%% Stage 2: FMM setup
% Box assignment
tic;
a_min = fmm_min_box_size(N_four, k2, tol_fmm);
G_target = max(16, ceil(sqrt(M))^2);
fmm = fmm_box_assign(centers, G_target, a_min);
t_info.t_boxes = toc;
fprintf('  Box assignment: %.2f s (grid %dx%d, box=%.2f)\n', ...
    t_info.t_boxes, fmm.Gx, fmm.Gy, fmm.box_size);

% FMM parameters
params = fmm_params(N_four, fmm.box_size, k2, tol_fmm);
fprintf('  P_tilde=%d, Q=%d\n', params.P_tilde, params.Q);

% Aggregation matrices
tic;
agg = fmm_aggregation(fmm, params, centers, N_four);
t_info.t_agg = toc;
fprintf('  Aggregation: %.2f s\n', t_info.t_agg);

% Translation precompute
tic;
fmm_trans = fmm_precompute(fmm, params, k2);
t_info.t_trans = toc;
fprintf('  Translation precompute: %.2f s\n', t_info.t_trans);

% Near-interaction matrix
tic;
near = fmm_near_matrix(fmm, centers, k2, N_four);
t_info.t_near = toc;
fprintf('  Near matrix: %.2f s (nnz=%d, fill=%.1f%%)\n', ...
    t_info.t_near, nnz(near.T_near), 100*nnz(near.T_near)/Ndof^2);

t_info.t_setup = t_info.t_Xblk + t_info.t_boxes + t_info.t_agg + ...
    t_info.t_trans + t_info.t_near;
fprintf('  Total setup: %.2f s\n', t_info.t_setup);

%% Stage 3: Right-hand side
rhs = X_blk * a_vec;

%% Stage 4: GMRES solve with FMM matvec
% Define the operator (I - X*T) as a function handle
afun = @(b) b - X_blk * fmm_matvec(b, fmm, agg, fmm_trans, near, N_four);

restart = min(Ndof, 100);
fprintf('  Starting GMRES (restart=%d, tol=%.1e, maxit=%d)...\n', ...
    restart, tol_gmres, maxit);

tic;
[beta_vec, flag, relres, iter] = gmres(afun, rhs, restart, tol_gmres, ...
    min(maxit, Ndof));
t_info.t_gmres = toc;

if flag == 0
    fprintf('  GMRES converged in %d iterations, %.2f s\n', iter(2), t_info.t_gmres);
else
    fprintf('  GMRES flag=%d, relres=%.2e, iter=%d, %.2f s\n', ...
        flag, relres, iter(2), t_info.t_gmres);
end
fprintf('  Relative residual: %.4e\n', relres);

t_info.t_total = t_info.t_setup + t_info.t_gmres;
fprintf('  Total time: %.2f s\n', t_info.t_total);

end
