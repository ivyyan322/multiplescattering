function fmm_trans = fmm_precompute(fmm, params, k2)
% FMM_PRECOMPUTE  Precompute FMM translation diagonals for all far box pairs.
%
%   fmm_trans = fmm_precompute(fmm, params, k2)
%
%   For each pair of far-interacting non-empty boxes (g, g'), precomputes
%   the Q x 1 diagonal translation vector F^(g',g) so it doesn't need
%   to be recomputed in each GMRES iteration.
%
%   Inputs:
%     fmm    - struct from fmm_box_assign
%     params - struct from fmm_params
%     k2     - exterior wavenumber
%
%   Output:
%     fmm_trans - struct with:
%       .F_diags  - G_total x G_total cell array: F_diags{g_target, g_source}
%                   is Q x 1 diagonal vector (empty if not a far pair)

G_total = fmm.G_total;
Q = params.Q;
P_tilde = params.P_tilde;
theta_q = params.theta_q;

F_diags = cell(G_total, G_total);

for gi = 1:length(fmm.nonempty_boxes)
    g_source = fmm.nonempty_boxes(gi);
    
    for fi = 1:length(fmm.far_list{g_source})
        g_target = fmm.far_list{g_source}(fi);
        
        % Translation vector: target - source (confirmed convention)
        x_vec = fmm.box_centers(g_target,:) - fmm.box_centers(g_source,:);
        x_dist = norm(x_vec);
        x_angle = atan2(x_vec(2), x_vec(1));
        
        F_d = zeros(Q, 1);
        for q = 1:Q
            val = 0;
            for xi = -P_tilde:P_tilde
                val = val + besselh(xi, 1, k2*x_dist) * ...
                    exp(1i * xi * (x_angle + pi/2 - theta_q(q)));
            end
            F_d(q) = val / Q;
        end
        
        F_diags{g_target, g_source} = F_d;
    end
end

fmm_trans.F_diags = F_diags;

end
