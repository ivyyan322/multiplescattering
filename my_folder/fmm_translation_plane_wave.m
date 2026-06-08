function F_diag = fmm_translation_plane_wave(params, k2, x_vec)
% FMM_TRANSLATION_PLANE_WAVE  Diagonal FMM translation matrix for a box pair.
%
%   F_diag = fmm_translation_plane_wave(params, k2, x_vec)
%
%   Computes the Q x Q diagonal FMM translation matrix F^(g,g') (Eq. 22)
%   for translating plane waves from box centered at the origin to
%   box centered at x_vec = c_g - c_{g'}.
%
%   F^(g,g')_{q,q} = (1/Q) * F_{P_tilde}(theta_q, x_vec)
%
%   where F_{P_tilde} is the FMM translation function (Eq. 19):
%     F_{P_tilde}(theta, x) = sum_{xi=-P_tilde}^{P_tilde}
%                              H^(1)_xi(k|x|) * exp(i*xi*(angle(x) + pi/2 - theta))
%
%   Inputs:
%     params - struct from fmm_params (contains P_tilde, Q, theta_q)
%     k2     - exterior wavenumber
%     x_vec  - 1 x 2 vector from target box center to source box center
%              (i.e., c_g - c_{g'} in the paper's notation)
%
%   Output:
%     F_diag - Q x 1 vector of diagonal entries (apply as F_diag .* v)

P_tilde = params.P_tilde;
Q = params.Q;
theta_q = params.theta_q;

% Distance and angle of translation vector
x_dist = norm(x_vec);
x_angle = atan2(x_vec(2), x_vec(1));

% Evaluate the FMM translation function at each quadrature angle
F_diag = zeros(Q, 1);

for q = 1:Q
    val = 0;
    for xi = -P_tilde:P_tilde
        H_xi = besselh(xi, 1, k2 * x_dist);
        val = val + H_xi * exp(1i * xi * (x_angle + pi/2 - theta_q(q)));
    end
    F_diag(q) = val / Q;
end

end
