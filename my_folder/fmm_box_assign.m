function fmm = fmm_box_assign(centers, G_target, min_box_size)
% FMM_BOX_ASSIGN  Partition particles into boxes and build near/far lists.
%
%   fmm = fmm_box_assign(centers, G_target, min_box_size)
%
%   Partitions M particles into a grid of approximately G_target boxes.
%   Builds near-interaction and far-interaction lists for each box pair.
%
%   Inputs:
%     centers      - M x 2 array of particle center coordinates
%     G_target     - target number of boxes (default: ceil(sqrt(M)))
%     min_box_size - minimum box side length (default: 0, no constraint).
%                    Use fmm_min_box_size() to compute this from FMM params.
%
%   Output:
%     fmm - struct with fields:
%       .M           - number of particles
%       .Gx, .Gy     - number of boxes in x and y directions
%       .G           - total number of non-empty boxes
%       .G_total     - Gx * Gy (total grid boxes including empty)
%       .box_size    - side length of each box
%       .box_origin  - [xmin, ymin] of the grid (lower-left corner)
%       .box_centers - G_total x 2 array of box center coordinates
%       .particle_box  - M x 1 array: box index for each particle
%       .box_particles - G_total x 1 cell array: particle indices in each box
%       .nonempty_boxes - vector of box indices that contain particles
%       .box_Mg      - G_total x 1: number of particles in each box
%       .near_list   - G_total x 1 cell: near-interaction box indices for each box
%       .far_list    - G_total x 1 cell: far-interaction box indices for each box
%       .is_near     - G_total x G_total logical: true if boxes are near

M = size(centers, 1);
if nargin < 2 || isempty(G_target)
    G_target = ceil(sqrt(M));
end
if nargin < 3 || isempty(min_box_size)
    min_box_size = 0;
end

%% Compute bounding box with padding
xmin = min(centers(:,1)); xmax = max(centers(:,1));
ymin = min(centers(:,2)); ymax = max(centers(:,2));

% Add small padding to avoid particles exactly on boundaries
Lx = xmax - xmin;
Ly = ymax - ymin;
pad = max(Lx, Ly) * 0.01 + 1e-10;  % 1% padding + epsilon
xmin = xmin - pad; xmax = xmax + pad;
ymin = ymin - pad; ymax = ymax + pad;
Lx = xmax - xmin;
Ly = ymax - ymin;

%% Determine grid dimensions
% Make boxes approximately square
% Total area = Lx * Ly, want G_target boxes of area a^2
% So a = sqrt(Lx * Ly / G_target)
a = sqrt(Lx * Ly / G_target);

% Enforce minimum box size
a = max(a, min_box_size);

% Compute grid counts from desired box size
Gx = max(1, floor(Lx / a));
Gy = max(1, floor(Ly / a));

% Final box size: must tile the domain AND satisfy the minimum
box_size = max([Lx / Gx, Ly / Gy, min_box_size]);

% Recompute grid counts with the final box size
Gx = max(1, floor(Lx / box_size));
Gy = max(1, floor(Ly / box_size));

% Final adjustment: ensure box_size * G >= L
box_size = max(Lx / Gx, Ly / Gy);
box_size = max(box_size, min_box_size);

% Adjust domain to fit exactly Gx * Gy square boxes
Lx_new = Gx * box_size;
Ly_new = Gy * box_size;
% Center the new domain on the old one
x_shift = (Lx_new - Lx) / 2;
y_shift = (Ly_new - Ly) / 2;
xmin = xmin - x_shift;
ymin = ymin - y_shift;

G_total = Gx * Gy;

%% Assign particles to boxes
% Box (ix, iy) has linear index (iy-1)*Gx + ix, ix=1..Gx, iy=1..Gy
particle_box = zeros(M, 1);
for m = 1:M
    ix = floor((centers(m,1) - xmin) / box_size) + 1;
    iy = floor((centers(m,2) - ymin) / box_size) + 1;
    % Clamp to grid (safety for edge cases)
    ix = max(1, min(Gx, ix));
    iy = max(1, min(Gy, iy));
    particle_box(m) = (iy-1)*Gx + ix;
end

%% Build box_particles cell array
box_particles = cell(G_total, 1);
box_Mg = zeros(G_total, 1);
for g = 1:G_total
    box_particles{g} = find(particle_box == g);
    box_Mg(g) = length(box_particles{g});
end

nonempty_boxes = find(box_Mg > 0);
G = length(nonempty_boxes);

%% Compute box centers
box_centers = zeros(G_total, 2);
for g = 1:G_total
    iy = ceil(g / Gx);
    ix = g - (iy-1)*Gx;
    box_centers(g,:) = [xmin + (ix-0.5)*box_size, ymin + (iy-0.5)*box_size];
end

%% Build near/far lists
% Two boxes are "near" if they are the same box or adjacent (including diagonal).
% In grid coordinates, this means |ix1 - ix2| <= 1 AND |iy1 - iy2| <= 1.
% Only consider non-empty boxes.

is_near = false(G_total);
near_list = cell(G_total, 1);
far_list = cell(G_total, 1);

for g1 = 1:G_total
    iy1 = ceil(g1 / Gx);
    ix1 = g1 - (iy1-1)*Gx;
    
    near_g1 = [];
    far_g1 = [];
    
    for g2 = 1:G_total
        if box_Mg(g2) == 0; continue; end  % skip empty boxes
        
        iy2 = ceil(g2 / Gx);
        ix2 = g2 - (iy2-1)*Gx;
        
        if abs(ix1 - ix2) <= 1 && abs(iy1 - iy2) <= 1
            is_near(g1, g2) = true;
            if g1 ~= g2  % don't include self in near list
                near_g1(end+1) = g2;
            end
        else
            far_g1(end+1) = g2;
        end
    end
    
    near_list{g1} = near_g1;
    far_list{g1} = far_g1;
end

%% Pack output struct
fmm.M = M;
fmm.Gx = Gx;
fmm.Gy = Gy;
fmm.G = G;
fmm.G_total = G_total;
fmm.box_size = box_size;
fmm.box_origin = [xmin, ymin];
fmm.box_centers = box_centers;
fmm.particle_box = particle_box;
fmm.box_particles = box_particles;
fmm.nonempty_boxes = nonempty_boxes;
fmm.box_Mg = box_Mg;
fmm.near_list = near_list;
fmm.far_list = far_list;
fmm.is_near = is_near;

end
