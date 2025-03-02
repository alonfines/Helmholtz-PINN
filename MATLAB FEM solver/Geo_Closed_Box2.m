% SET GEOMETRY & MESH 

%% ---------- Define the geometry
% --> Quadrangle format 1 x 10: [3 4 x_UpLeft x_UpR x_LoR x_LoLeft y_UpLeft y_UpR y_LoR y_LoLeft]
% --> Circle format 1 x 4: [1 x_c y_c radius]

Omega   = [3 4 0 1 1 0 1 1 0 0];  % Square domain                   % Unit square domain
Antenna = [1 0.5 0 0.005 zeros(1,6)]; % Source at (0.5, 0)


%% ---------- Define the computational domainand build the mesh
Domain = [Omega; Antenna]';
ns     = (char('Tr','C1'))'; % Names of objects
sf     = 'Tr-C1';            % Omega minus the source and the obstacle (hole)

[mesh,model] = fun_get_Mesh(Domain,sf,ns,H_max,H_min); % Assemble the geometry and build the mesh
% triplot(mesh.T); % Have a look on the mesh
% pdegplot(model,'EdgeLabels','on'); 
% --> Useful to check the edge labels to apply boundary conditions
% --> Another mesh file (for instance from GMSH) can be loaded, to skip all the code
%     until now. Just needs to set the good PET format.


%% ---------- Set the boundary conditions on the expected edges
mesh.e_Dir = [6 : 7]; % Labels of edges supporting inhom Dir cond
mesh.e_ABC = [1:5];      % Labels of transparent edges
% --> Use pdegplot above to check edges labels.
% --> Here: apply 1 on Antenna (edges 5 to 8), hom Dir cond on Omega (multi-reflexions).
% --> By default the non specified edges support hom Dir cond.

figure;
pdegplot(model, 'EdgeLabels', 'on');
title('Geometry with Edge Labels');