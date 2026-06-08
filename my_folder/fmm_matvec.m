function y = fmm_matvec(beta, fmm, agg, fmm_trans, near, N_four)
% FMM_MATVEC  Compute y = T * beta using FMM acceleration.
%
%   y = fmm_matvec(beta, fmm, agg, fmm_trans, near, N_four)
%
%   Replaces the dense matrix-vector product T*beta with:
%     y = y_near + y_far
%   where y_near uses the sparse near-interaction matrix and y_far uses
%   the FMM aggregation/translation/disaggregation pipeline.
%
%   Algorithm:
%     1. y_near = near.T_near * beta
%     2. For each source box g, aggregate: pw_g = sum_m A^(m) * beta^(m)
%     3. For each (target, source) far pair, translate: pw_target += F * pw_source
%     4. For each target particle l, disaggregate: y_far(l) = A^(l)' * pw_target_box
%
%   Inputs:
%     beta      - (Nmodes*M x 1) coefficient vector
%     fmm       - struct from fmm_box_assign
%     agg       - struct from fmm_aggregation
%     fmm_trans - struct from fmm_precompute
%     near      - struct from fmm_near_matrix
%     N_four    - cylindrical harmonics truncation
%
%   Output:
%     y - (Nmodes*M x 1) result of T * beta

M = fmm.M;
Nmodes = 2*N_four + 1;
Q = agg.Q;
G_total = fmm.G_total;

%% Step 1: Near interactions (sparse matvec)
y_near = near.T_near * beta;

%% Step 2: Aggregation — for each source box, aggregate particle contributions
% pw_boxes{g} = sum of A^(m) * beta^(m) for all particles m in box g
pw_boxes = cell(G_total, 1);

for gi = 1:length(fmm.nonempty_boxes)
    g = fmm.nonempty_boxes(gi);
    particles_g = fmm.box_particles{g};
    
    pw_g = zeros(Q, 1);
    for pi = 1:length(particles_g)
        m = particles_g(pi);
        beta_m = beta((m-1)*Nmodes + (1:Nmodes));
        pw_g = pw_g + agg.A_particles{m} * beta_m;
    end
    pw_boxes{g} = pw_g;
end

%% Step 3: FMM translation — for each target box, accumulate translated plane waves
pw_incoming = cell(G_total, 1);
for gi = 1:length(fmm.nonempty_boxes)
    g_target = fmm.nonempty_boxes(gi);
    pw_incoming{g_target} = zeros(Q, 1);
end

for gi = 1:length(fmm.nonempty_boxes)
    g_source = fmm.nonempty_boxes(gi);
    
    for fi = 1:length(fmm.far_list{g_source})
        g_target = fmm.far_list{g_source}(fi);
        
        F_d = fmm_trans.F_diags{g_target, g_source};
        pw_incoming{g_target} = pw_incoming{g_target} + F_d .* pw_boxes{g_source};
    end
end

%% Step 4: Disaggregation — project incoming plane waves onto each target particle
y_far = zeros(Nmodes*M, 1);

for gi = 1:length(fmm.nonempty_boxes)
    g = fmm.nonempty_boxes(gi);
    
    if isempty(fmm.far_list{g}) || all(cellfun(@isempty, fmm_trans.F_diags(g, :)))
        continue;  % no far contributions to this box
    end
    
    particles_g = fmm.box_particles{g};
    pw_in = pw_incoming{g};
    
    for pi = 1:length(particles_g)
        l = particles_g(pi);
        % Disaggregation: A^(l)' * pw_incoming
        y_l = agg.A_particles{l}' * pw_in;
        y_far((l-1)*Nmodes + (1:Nmodes)) = y_l;
    end
end

%% Combine
y = y_near + y_far;

end
