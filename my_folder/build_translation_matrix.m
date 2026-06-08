function T_full = build_translation_matrix(centers, k2, N_four)
% BUILD_TRANSLATION_MATRIX  Assemble M2L translation matrix for M particles.
%
%   T_full = build_translation_matrix(centers, k2, N_four)
%
%   Builds the full (Nmodes*M) x (Nmodes*M) translation matrix T, where
%   Nmodes = 2*N_four+1 and M = size(centers,1).
%
%   T(m',m) translates outgoing coefficients of particle m into incoming
%   coefficients at particle m', using Graf's addition theorem:
%     T(m',m)_{mu,p} = exp(i*(p-mu)*theta_{m,m'}) * H^(1)_{p-mu}(k2*d_{m,m'})
%
%   Note: the sign convention follows Eq. (13) of Blankrot & Heitzinger 2019,
%   where the translation is from m to m'.
%
%   Inputs:
%     centers - M x 2 array of particle center coordinates
%     k2      - exterior wavenumber
%     N_four  - cylindrical harmonics truncation parameter
%
%   Output:
%     T_full  - (Nmodes*M) x (Nmodes*M) translation matrix

M = size(centers, 1);
Nmodes = 2*N_four + 1;
nvec = (-N_four:N_four)';
T_full = zeros(Nmodes*M);

for l = 1:M
    for m = 1:M
        if l == m; continue; end
        
        % Vector from target l to source m (matches original code convention)
        d_vec = centers(m,:) - centers(l,:);
        d = norm(d_vec);
        theta_lm = atan2(d_vec(2), d_vec(1));
        
        Tlm = zeros(Nmodes);
        for idx_mu = 1:Nmodes
            mu = nvec(idx_mu);
            for idx_p = 1:Nmodes
                p = nvec(idx_p);
                nn = mu - p;
                Tlm(idx_mu, idx_p) = exp(-1i*nn*theta_lm) * besselh(nn, 1, k2*d);
            end
        end
        
        T_full((l-1)*Nmodes+(1:Nmodes), (m-1)*Nmodes+(1:Nmodes)) = Tlm;
    end
end
end
