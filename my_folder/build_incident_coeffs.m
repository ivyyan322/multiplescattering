function a_vec = build_incident_coeffs(centers, k2, N_four, theta_inc)
% BUILD_INCIDENT_COEFFS  Incident plane wave coefficients for each particle.
%
%   a_vec = build_incident_coeffs(centers, k2, N_four, theta_inc)
%
%   For a plane wave u_inc = exp(i*k2*(x*cos(theta_inc) + y*sin(theta_inc))),
%   computes the local cylindrical harmonic expansion coefficients at each
%   particle center using the Jacobi-Anger expansion:
%     alpha_n^(m) = i^n * exp(-i*n*theta_inc) * exp(i*k2*d_inc . o^(m))
%
%   where d_inc = (cos(theta_inc), sin(theta_inc)) is the incident direction.
%
%   Inputs:
%     centers   - M x 2 array of particle center coordinates
%     k2        - exterior wavenumber
%     N_four    - cylindrical harmonics truncation
%     theta_inc - angle of incidence (radians), default 0 (x-traveling)
%
%   Output:
%     a_vec     - (Nmodes*M) x 1 vector of incident coefficients

if nargin < 4
    theta_inc = 0;  % default: plane wave traveling in +x direction
end

M = size(centers, 1);
Nmodes = 2*N_four + 1;
nvec = (-N_four:N_four)';
a_vec = zeros(Nmodes*M, 1);

d_inc = [cos(theta_inc), sin(theta_inc)];

for m = 1:M
    % Phase shift for particle center location
    phase = exp(1i * k2 * dot(d_inc, centers(m,:)));
    
    for idx = 1:Nmodes
        n = nvec(idx);
        a_vec((m-1)*Nmodes + idx) = 1i^n * exp(-1i*n*theta_inc) * phase;
    end
end
end
