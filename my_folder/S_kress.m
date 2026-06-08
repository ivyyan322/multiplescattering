function [Sk] = S_kress(k, x, y, dx, dy, ddx, ddy, ds, h)
% S_KRESS  Single-layer potential operator with Kress quadrature.
%
%   Sk = S_kress(k, x, y, dx, dy, ddx, ddy, ds, h)
%
%   Assembles the Nd x Nd matrix for the single-layer potential operator
%   S_k on a smooth closed boundary, using Kussmaul-Martensen quadrature
%   to handle the logarithmic singularity of the 2D Helmholtz Green's function.

M = length(x);
t_param = (0:M-1)' * h;
R = kress_weights(M);
gamma_E = 0.5772156649015329;
Sk = zeros(M);

for i = 1:M
    for j = 1:M
        rx = x(i)-x(j); ry = y(i)-y(j);
        r = sqrt(rx^2+ry^2);
        ell = mod(j-i+M,M);
        if ell > M/2; ell = M-ell; end

        if i == j
            phi_ij = -(1/(4*pi)) * ds(j);
            psi_ij = ds(j) * (-(1/(2*pi))*(log(k*ds(j)/2) + gamma_E) + 1i/4);
            Sk(i,j) = R(ell+1)*phi_ij + h*psi_ij;
        else
            phi_ij = -(1/(4*pi)) * besselj(0,k*r) * ds(j);
            K_full = (1i/4) * besselh(0,1,k*r) * ds(j);
            dt = t_param(i)-t_param(j);
            L_ij = log(4*sin(dt/2)^2);
            psi_ij = K_full - phi_ij*L_ij;
            Sk(i,j) = R(ell+1)*phi_ij + h*psi_ij;
        end
    end
end
end
