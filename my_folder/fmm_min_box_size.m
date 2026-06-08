function a_min = fmm_min_box_size(N_four, k2, tol)
% FMM_MIN_BOX_SIZE  Minimum box size for FMM feasibility.
%
%   Need: 2P + k2*a*sqrt(2) + digits < k2*2*a
%   So:   2P + digits < k2*a*(2 - sqrt(2))
%   Hence: a > (2P + digits) / (k2 * (2-sqrt(2)))

P = N_four;
digits = -log10(tol);
a_min = (2*P + digits) / (k2 * (2 - sqrt(2)));

end