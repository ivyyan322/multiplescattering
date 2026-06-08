function agg = fmm_aggregation(fmm, params, centers, N_four)
% FMM_AGGREGATION  Build aggregation matrices for all particles.
%
%   agg = fmm_aggregation(fmm, params, centers, N_four)
%
%   For each particle m in box g, builds the Q x Nmodes aggregation matrix
%   A^(m) (Eq. 21 in Blankrot & Heitzinger 2019):
%
%     A^(m)_{q,n} = exp(-i * k_q . (o^(m) - c_g)) * exp(-i*n*(pi/2 - theta_q))
%
%   where k_q = k*(cos(theta_q), sin(theta_q)), o^(m) is the particle center,
%   c_g is the box center, and n runs from -N_four to N_four.
%
%   The disaggregation matrix for each particle is (1/Q) * A^(m)^H,
%   applied at the target box. Since D_g = A_g^H in the paper's notation
%   (Eq. 32, Appendix), the 1/Q factor comes from the quadrature weight.
%
%   Inputs:
%     fmm     - struct from fmm_box_assign
%     params  - struct from fmm_params
%     centers - M x 2 particle centers
%     N_four  - cylindrical harmonics truncation
%
%   Output:
%     agg - struct with fields:
%       .A_particles  - M x 1 cell array: A^(m) is Q x Nmodes for each particle
%       .Q            - number of quadrature points
%       .Nmodes       - 2*N_four + 1

M = fmm.M;
Q = params.Q;
Nmodes = 2*N_four + 1;
nvec = (-N_four:N_four);
theta_q = params.theta_q;
kq = params.kq;

A_particles = cell(M, 1);

for m = 1:M
    g = fmm.particle_box(m);
    cg = fmm.box_centers(g, :);
    om = centers(m, :);
    
    % Displacement from box center to particle
    d_shift = om - cg;
    
    A_m = zeros(Q, Nmodes);
    for q = 1:Q
        % Phase from particle offset within box
        phase_shift = exp(-1i * (kq(q,:) * d_shift'));
        
        for idx = 1:Nmodes
            n = nvec(idx);
            % Angular phase
            phase_angle = exp(-1i * n * (pi/2 - theta_q(q)));
            
            A_m(q, idx) = phase_shift * phase_angle;
        end
    end
    
    A_particles{m} = A_m;
end

agg.A_particles = A_particles;
agg.Q = Q;
agg.Nmodes = Nmodes;

end
