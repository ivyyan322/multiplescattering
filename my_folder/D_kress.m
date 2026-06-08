function [Dk] = D_kress(k, x, y, dx, dy, ddx, ddy, ds, h)
% D_KRESS  Double-layer potential operator with Kress quadrature.
%
%   Dk = D_kress(k, x, y, dx, dy, ddx, ddy, ds, h)
%
%   Assembles the Nd x Nd matrix for the double-layer potential operator
%   D_k on a smooth closed boundary.

M = length(x);
nx = dy./ds; ny = -dx./ds;
t_param = (0:M-1)' * h;
R = kress_weights(M);
kappa = (dx.*ddy - dy.*ddx) ./ ds.^3;
Dk = zeros(M);

for i = 1:M
    for j = 1:M
        rx = x(i)-x(j); ry = y(i)-y(j);
        r = sqrt(rx^2+ry^2);
        ell = mod(j-i+M,M);
        if ell > M/2; ell = M-ell; end

        if i == j
            phi_ij = 0;
            psi_ij = (1/(4*pi)) * kappa(j) * ds(j);
            Dk(i,j) = R(ell+1)*phi_ij + h*psi_ij;
        else
            rdotn_j = rx*nx(j)+ry*ny(j);
            phi_ij = (1/2)*(-(1/(2*pi))*k*besselj(1,k*r)*rdotn_j/r)*ds(j);
            K_full = (1i*k/4)*besselh(1,1,k*r)*rdotn_j/r*ds(j);
            dt = t_param(i)-t_param(j);
            L_ij = log(4*sin(dt/2)^2);
            psi_ij = K_full - phi_ij*L_ij;
            Dk(i,j) = R(ell+1)*phi_ij + h*psi_ij;
        end
    end
end
end
