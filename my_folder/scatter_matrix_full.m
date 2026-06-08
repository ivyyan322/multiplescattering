function X = scatter_matrix_full(k2, kp, x, y, N_four, dx, dy, ddx, ddy)
% SCATTER_MATRIX_FULL  Compute scattering matrix for a penetrable inclusion.
%
%   X = scatter_matrix_full(k2, kp, x, y, N_four, dx, dy, ddx, ddy)
%
%   Builds the (2*N_four+1) x (2*N_four+1) scattering matrix that maps
%   incoming cylindrical harmonic coefficients alpha to outgoing coefficients
%   beta, for a penetrable inclusion with exterior wavenumber k2 and
%   interior wavenumber kp.
%
%   The inclusion boundary is parameterized by (x(t), y(t)) centered at
%   the origin, with derivatives (dx, dy), (ddx, ddy), and arc length ds.
%
%   Each column of X is computed by solving the combined-field BIE system
%   for a cylindrical harmonic incident wave J_n(k2*r)*exp(i*n*theta),
%   then projecting the resulting potentials onto outgoing Hankel harmonics.

X = zeros(2*N_four+1);
nvec_loc = -N_four:N_four;
for idx = 1:length(nvec_loc)
    l = nvec_loc(idx);
    b_ln = scattering_An_arb(l, k2, kp, x, y, N_four, dx, dy, ddx, ddy);
    X(idx,:) = b_ln;
end
end
