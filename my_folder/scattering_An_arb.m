function [b_ln] = scattering_An_arb(l, k2, kp, x, y, N, dx, dy, ddx, ddy)
% SCATTERING_AN_ARB  Compute one row of the scattering matrix.
%
%   b_ln = scattering_An_arb(l, k2, kp, x, y, N, dx, dy, ddx, ddy)
%
%   For outgoing harmonic index l, solves the BIE system for each incoming
%   harmonic n = -N,...,N and projects onto the l-th outgoing Hankel harmonic
%   via the integral formula (Eq. 7 in Blankrot & Heitzinger 2019).

M = length(x);
ds = sqrt(dx.^2 + dy.^2);
r = sqrt(x.^2 + y.^2);
h = 2*pi/M;
nx = dy./ds; ny = -dx./ds;
theta = atan2(y, x);

% Build the BIE system matrix (same for all incident waves)
Dk2 = D_kress(k2, x, y, dx, dy, ddx, ddy, ds, h);
Dkp = D_kress(kp, x, y, dx, dy, ddx, ddy, ds, h);
Sk2 = S_kress(k2, x, y, dx, dy, ddx, ddy, ds, h);
Skp = S_kress(kp, x, y, dx, dy, ddx, ddy, ds, h);
Nk2 = N_kress(k2, x, y, dx, dy, ddx, ddy, ds, h);
Nkp = N_kress(kp, x, y, dx, dy, ddx, ddy, ds, h);
Tdiff = T_diff_kress(k2, kp, x, y, dx, dy, ddx, ddy, ds, h);

A = [(Dk2-Dkp)+eye(M), (Sk2-Skp); Tdiff, (Nk2-Nkp)-eye(M)];

% Precompute the test function v_l = J_l(k2*r)*exp(-i*l*theta)
Jl = besselj(l, k2*r);
Jlp = 0.5*(besselj(l-1, k2*r) - besselj(l+1, k2*r));
v_l = Jl .* exp(-1i*l*theta);
dvdr = k2*Jlp .* exp(-1i*l*theta);
dvdtheta = -1i*l*Jl .* exp(-1i*l*theta);
gx = dvdr.*cos(theta) - dvdtheta.*sin(theta)./r;
gy = dvdr.*sin(theta) + dvdtheta.*cos(theta)./r;
ndotgradv = gx.*nx + gy.*ny;

% LU factorize A for repeated solves
[L_fac, U_fac, P_fac] = lu(A);

b_ln = zeros(2*N+1, 1);
for idx = 1:(2*N+1)
    n = idx - N - 1;
    
    % Incident wave: J_n(k2*r)*exp(i*n*theta)
    Jn = besselj(n, k2*r);
    Jnp = 0.5*(besselj(n-1, k2*r) - besselj(n+1, k2*r));
    uinc_n = Jn .* exp(1i*n*theta);
    dudr_n = k2*Jnp .* exp(1i*n*theta);
    dudtheta_n = 1i*n*Jn .* exp(1i*n*theta);
    gx_n = dudr_n.*cos(theta) - dudtheta_n.*sin(theta)./r;
    gy_n = dudr_n.*sin(theta) + dudtheta_n.*cos(theta)./r;
    duinc_n_dn = gx_n.*nx + gy_n.*ny;
    
    % Solve BIE system
    sol = U_fac \ (L_fac \ (P_fac * [-uinc_n; -duinc_n_dn]));
    mu_n = sol(1:M);
    sigma_n = sol(M+1:2*M);
    
    % Project onto outgoing harmonic l
    b_ln(idx) = (1i/4) * sum((v_l.*sigma_n + ndotgradv.*mu_n) .* ds * h);
end
end
