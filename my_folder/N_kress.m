function [Nk] = N_kress(k, x, y, dx, dy, ddx, ddy, ds, h)
% N_KRESS  Normal derivative of single-layer potential operator (Kress).
%
%   Nk = N_kress(k, x, y, dx, dy, ddx, ddy, ds, h)
%
%   Assembles the Nd x Nd matrix for the operator d/dn_x S_k,
%   i.e., the normal derivative of the single-layer potential at target points.

M = length(x);
nx = dy./ds; ny = -dx./ds;
t_param = (0:M-1)' * h;
R = kress_weights(M);
kappa = (dx.*ddy - dy.*ddx) ./ ds.^3;
Nk = zeros(M);

for i = 1:M
    for j = 1:M
        rx = x(i)-x(j); ry = y(i)-y(j);
        r = sqrt(rx^2+ry^2);
        ell = mod(j-i+M,M);
        if ell > M/2; ell = M-ell; end

        if i == j
            phi_ij = 0;
            psi_ij = -(1/(4*pi)) * kappa(i) * ds(j);
            Nk(i,j) = R(ell+1)*phi_ij + h*psi_ij;
        else
            rdotn_i = rx*nx(i)+ry*ny(i);
            phi_ij = (k/(4*pi))*besselj(1,k*r)*rdotn_i/r*ds(j);
            K_full = -(1i*k/4)*besselh(1,1,k*r)*rdotn_i/r*ds(j);
            dt = t_param(i)-t_param(j);
            L_ij = log(4*sin(dt/2)^2);
            psi_ij = K_full - phi_ij*L_ij;
            Nk(i,j) = R(ell+1)*phi_ij + h*psi_ij;
        end
    end
end
end
