function [u_tot, u_sc, u_inc, X_grid, Y_grid] = compute_field_grid(...
    beta_vec, centers, rotations, k2, N_four, theta_inc, ...
    x_range, y_range, Nx, Ny, ac, bc)
% COMPUTE_FIELD_GRID  Evaluate total field on a rectangular grid.
%
%   [u_tot, u_sc, u_inc, X_grid, Y_grid] = compute_field_grid(...)
%
%   Evaluates the scattered field via cylindrical harmonic expansion and
%   adds the incident plane wave. Points inside any scatterer are masked
%   as NaN for clean visualization.
%
%   Inputs:
%     beta_vec   - (Nmodes*M x 1) outgoing coefficients
%     centers    - M x 2 particle centers
%     rotations  - M x 1 rotation angles
%     k2         - exterior wavenumber
%     N_four     - cylindrical harmonics truncation
%     theta_inc  - incidence angle (radians)
%     x_range    - [xmin, xmax] for grid
%     y_range    - [ymin, ymax] for grid
%     Nx, Ny     - number of grid points in x and y
%     ac, bc     - ellipse semi-axes (for masking interior points)
%
%   Outputs:
%     u_tot   - Ny x Nx complex total field (NaN inside scatterers)
%     u_sc    - Ny x Nx complex scattered field
%     u_inc   - Ny x Nx complex incident field
%     X_grid  - Ny x Nx meshgrid x-coordinates
%     Y_grid  - Ny x Nx meshgrid y-coordinates

M = size(centers, 1);
Nmodes = 2*N_four + 1;
nvec = (-N_four:N_four)';

% Build evaluation grid
xv = linspace(x_range(1), x_range(2), Nx);
yv = linspace(y_range(1), y_range(2), Ny);
[X_grid, Y_grid] = meshgrid(xv, yv);

% Flatten for evaluation
eval_pts = [X_grid(:), Y_grid(:)];
Npts = size(eval_pts, 1);

% Evaluate scattered field at all points
u_sc_flat = zeros(Npts, 1);
for pt = 1:Npts
    xo = eval_pts(pt, 1);
    yo = eval_pts(pt, 2);
    
    for m = 1:M
        rm = sqrt((xo - centers(m,1))^2 + (yo - centers(m,2))^2);
        thetam = atan2(yo - centers(m,2), xo - centers(m,1));
        beta_m = beta_vec((m-1)*Nmodes + (1:Nmodes));
        
        for idx = 1:Nmodes
            n = nvec(idx);
            u_sc_flat(pt) = u_sc_flat(pt) + ...
                beta_m(idx) * besselh(n, 1, k2*rm) * exp(1i*n*thetam);
        end
    end
end

% Incident field
d_inc = [cos(theta_inc), sin(theta_inc)];
u_inc_flat = exp(1i * k2 * (eval_pts(:,1)*d_inc(1) + eval_pts(:,2)*d_inc(2)));

% Total field
u_tot_flat = u_sc_flat + u_inc_flat;

% Mask points inside scatterers
for m = 1:M
    phi = rotations(m);
    R = [cos(-phi), -sin(-phi); sin(-phi), cos(-phi)];
    
    for pt = 1:Npts
        dx = eval_pts(pt,1) - centers(m,1);
        dy = eval_pts(pt,2) - centers(m,2);
        
        % Rotate to local frame
        local = R * [dx; dy];
        
        % Check if inside ellipse
        if (local(1)/ac)^2 + (local(2)/bc)^2 < 1
            u_tot_flat(pt) = NaN;
            u_sc_flat(pt) = NaN;
            u_inc_flat(pt) = NaN;
        end
    end
end

% Reshape to grid
u_tot = reshape(u_tot_flat, Ny, Nx);
u_sc = reshape(u_sc_flat, Ny, Nx);
u_inc = reshape(u_inc_flat, Ny, Nx);

end
