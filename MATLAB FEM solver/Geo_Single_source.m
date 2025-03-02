% SET GEOMETRY & MESH 

%% ---------- Define the geometry
% --> Quadrangle format 1 x 10: [3 4 x_UpLeft x_UpR x_LoR x_LoLeft y_UpLeft y_UpR y_LoR y_LoLeft]
% --> Circle format 1 x 4: [1 x_c y_c radius]

Omega   = [1 0 0 0.5];     % Unit disk domain
Antenna = [1 0 0.2 0.005]; % Small circle of radius 0.005 for source term


%% ---------- Define the computational domainand build the mesh
Domain = [Omega; Antenna]';
ns     = (char('C1','C2'))'; % Names of objects
sf     = 'C1-C2';            % Omega minus the source (hole)

[mesh,model] = fun_get_Mesh(Domain,sf,ns,H_max,H_min); % Assemble the geometry and build the mesh
% triplot(mesh.T); % Have a look on the mesh
% pdegplot(model,'EdgeLabels','on'); 
% --> Useful to check the edge labels to apply boundary conditions
% --> Another mesh file (for instance from GMSH) can be loaded, to skip all the code
%     until now. Just needs to set the good PET format.


%% ---------- Set the boundary conditions on the expected edges
mesh.e_Dir = [5 : 8]; % Labels of edges supporting inhom Dir cond
mesh.e_ABC = [1 : 4]; % Labels of transparent edges
% --> Use pdegplot above to check edges labels.
% --> Here: apply 1 on Antenna (edges 5 to 8), ABC on Omega (edges 1 to 4).
% --> By default the non specified edges support hom Dir cond.