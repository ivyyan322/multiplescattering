function near = fmm_near_matrix(fmm, centers, k2, N_four)
% FMM_NEAR_MATRIX  Assemble near-interaction translation blocks.
%
%   near = fmm_near_matrix(fmm, centers, k2, N_four)
%
%   For each pair of particles (m, l) where m ~= l and their boxes are
%   near-interacting (same box or adjacent), assembles the direct
%   translation block T^(l,m) and stores it for sparse matvec.
%
%   Inputs:
%     fmm     - struct from fmm_box_assign
%     centers - M x 2 particle centers
%     k2      - exterior wavenumber
%     N_four  - cylindrical harmonics truncation
%
%   Output:
%     near - struct with:
%       .T_near  - (Nmodes*M) x (Nmodes*M) sparse matrix containing
%                  only near-interaction blocks

M = fmm.M;
Nmodes = 2*N_four + 1;
nvec = (-N_four:N_four)';

% Preallocate sparse matrix using triplets
% Estimate nnz: for each near pair, Nmodes^2 entries
% Count near pairs first
n_near_pairs = 0;
for l = 1:M
    g_l = fmm.particle_box(l);
    % Same-box particles
    same_box = fmm.box_particles{g_l};
    n_near_pairs = n_near_pairs + length(same_box) - 1;  % exclude self
    % Adjacent-box particles
    for ni = 1:length(fmm.near_list{g_l})
        g_adj = fmm.near_list{g_l}(ni);
        n_near_pairs = n_near_pairs + length(fmm.box_particles{g_adj});
    end
end

nnz_est = n_near_pairs * Nmodes^2;
row_idx = zeros(nnz_est, 1);
col_idx = zeros(nnz_est, 1);
vals = zeros(nnz_est, 1);
count = 0;

for l = 1:M
    g_l = fmm.particle_box(l);
    
    % Collect all near-interacting source particles for target l
    source_particles = [];
    
    % Same box (exclude self)
    same_box = fmm.box_particles{g_l};
    source_particles = [source_particles; same_box(same_box ~= l)];
    
    % Adjacent boxes
    for ni = 1:length(fmm.near_list{g_l})
        g_adj = fmm.near_list{g_l}(ni);
        source_particles = [source_particles; fmm.box_particles{g_adj}];
    end
    
    for si = 1:length(source_particles)
        m = source_particles(si);
        
        % Compute T^(l,m) block
        d_vec = centers(m,:) - centers(l,:);
        d = norm(d_vec);
        theta_lm = atan2(d_vec(2), d_vec(1));
        
        for idx_mu = 1:Nmodes
            mu = nvec(idx_mu);
            row = (l-1)*Nmodes + idx_mu;
            
            for idx_p = 1:Nmodes
                p = nvec(idx_p);
                col = (m-1)*Nmodes + idx_p;
                nn = mu - p;
                
                count = count + 1;
                row_idx(count) = row;
                col_idx(count) = col;
                vals(count) = exp(-1i*nn*theta_lm) * besselh(nn, 1, k2*d);
            end
        end
    end
end

% Trim preallocated arrays
row_idx = row_idx(1:count);
col_idx = col_idx(1:count);
vals = vals(1:count);

near.T_near = sparse(row_idx, col_idx, vals, Nmodes*M, Nmodes*M);

end
