function R = kress_weights(M)
% KRESS_WEIGHTS  Compute Kress quadrature weights for log-singular integrals.
%
%   R = kress_weights(M)
%
%   Returns M weights for the Kussmaul-Martensen quadrature rule used
%   to handle logarithmic singularities in boundary integral operators.

R = zeros(M, 1);
for j = 0:M-1
    val = 0;
    for n = 1:M/2-1
        val = val + (1/n) * cos(n * 2*pi*j/M);
    end
    val = val + (1/M) * cos(pi*j);
    R(j+1) = -(4*pi/M) * val;
end
end
