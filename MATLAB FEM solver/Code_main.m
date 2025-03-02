%--------------------------------------------------------------------------
% FEM solver for 2D homogeneous Helmholtz equation on arbitrary geometry :
%                            Delta U + k^2 * U = 0.
% On the oundaries where we wish to apply transparent condition ABC, given  
% n the outwardly normal vector of the domain, the low order absorbing 
% condition (if expected) writes:
%                            n.grad U + ik * U = 0.
% P1 triangular elements.
%--------------------------------------------------------------------------
% Author: David Gasperini
% Created: 2021-05-05
% Reference:
%     [1] Cuvelier, F., Japhet, C., & Scarella, G. (2013). An efficient way
%         to perform the assembly of finite element matrices in Matlab and 
%         Octave. arXiv preprint arXiv:1305.3122.
%--------------------------------------------------------------------------


close all
%% ----------- Physical parameters
Amp_source = 2;     % Value(s) of inhom Dir cond (sources)
k = 100;             % Wavenumber
wave_length = 2*pi/k; % FYI, just to adjust the mesh element size (control P1 convergence)


%% ----------- Set geometry & mesh
H_max = 0.015;   H_min = 0.01; % Max and min bounds for elements sizes
run('Geo_Diffraction2');         % Choose ready-to-use geometry Geo_...
% --> Another mesh file (for instance from GMSH) can be loaded. Just needs 
%     to set the good PET format.
% --> Extensions to multiple sources and/or inhomogeneous Helmholtz equation 
%     is straight forward.


%% ----------- FEM assembly & resolution
U = fun_FEM_Process(k,Amp_source,mesh);
% --> Hand-made vectorized FEM assembly.

% Extract x, y coordinates from mesh
x_values = mesh.p(:,1);  % X-coordinates of the grid
z_values = mesh.p(:,2);  % Y-coordinates of the grid
u_rvalues = real(U);      % Computed solution (wave amplitudes)
u_ivalues = imag(U);

% Save the grid data to a MAT file
save('grid_data.mat', 'x_values', 'z_values', 'u_rvalues','u_ivalues');

% Display confirmation message
disp('Grid data (x, y, z) saved to grid_data.mat');

%% ----------- Plot
set(0,'DefaultFigureColormap',jet()); 
trisurf(mesh.t(:,1:3), mesh.p(:,1), mesh.p(:,2),real(U),'facecolor','interp');
shading interp; 