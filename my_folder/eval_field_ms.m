function [u_sc, u_tot] = eval_field_ms(eval_pts, beta_vec, centers, ...
    k2, N_four, theta_inc)
% EVAL_FIELD_MS  Evaluate scattered and total fields from multiple scattering.
%
%   [u_sc, u_tot] = eval_field_ms(eval_pts, beta_vec, centers, k2, N_four, theta_inc)
%
%   Evaluates the scattered field at evaluation points using the cylindrical
%   harmonic expansion from each particle, and adds the incident plane wave
%   to get the total field.
%
%   Inputs:
%     eval_pts  - Npts x 2 array of evaluation point coordinates
%     beta_vec  - (Nmodes*M x 1) outgoing coefficients
%     centers   - M x 2 array of particle center coordinates
%     k2        - exterior wavenumber
%     N_four    - cylindrical harmonics truncation
%     theta_inc - angle of incidence (radians), default 0
%
%   Outputs:
%     u_sc  - Npts x 1 complex scattered field
%     u_tot - Npts x 1 complex total field (u_sc + u_inc)

if nargin < 6
    theta_inc = 0;
end

Npts = size(eval_pts, 1);
M = size(centers, 1);
Nmodes = 2*N_four + 1;
nvec = (-N_four:N_four)';

u_sc = zeros(Npts, 1);

for pt = 1:Npts
    xo = eval_pts(pt, 1);
    yo = eval_pts(pt, 2);
    
    for m = 1:M
        rm = sqrt((xo - centers(m,1))^2 + (yo - centers(m,2))^2);
        thetam = atan2(yo - centers(m,2), xo - centers(m,1));
        beta_m = beta_vec((m-1)*Nmodes + (1:Nmodes));
        
        for idx = 1:Nmodes
            n = nvec(idx);
            u_sc(pt) = u_sc(pt) + beta_m(idx) * besselh(n, 1, k2*rm) * exp(1i*n*thetam);
        end
    end
end

% Incident field
d_inc = [cos(theta_inc), sin(theta_inc)];
u_inc = exp(1i * k2 * (eval_pts(:,1)*d_inc(1) + eval_pts(:,2)*d_inc(2)));
u_tot = u_sc + u_inc;

end
