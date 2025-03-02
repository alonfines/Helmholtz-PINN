% SET GEOMETRY & MESH 

%% ---------- Define the geometry
% --> Quadrangle format 1 x 10: [3 4 x_UpLeft x_UpR x_LoR x_LoLeft y_UpLeft y_UpR y_LoR y_LoLeft]
% --> Circle format 1 x 4: [1 x_c y_c radius]

Omega   = [1 0.5 0.5 0.5 zeros(1,6)];                    % Unit square domain
Obst_1  = [3 4 -0.01 0.35 0.35 -0.01 0.6 0.6 0.58 0.58]; % Obstacle 1: thin rectangle
Obst_2  = [3 4 0.4 0.6 0.6 0.4 0.6 0.6 0.58 0.58];       % Obstacle 2: thin rectangle
Obst_3  = [3 4 0.65 0.999 0.999 0.65 0.6 0.6 0.58 0.58]; % Obstacle 3: thin rectangle
Antenna = [1 0.5 0.4 0.005 zeros(1,6)];                  % Small circle for source term


%% ---------- Define the computational domainand build the mesh
Domain = [Omega; Obst_1; Obst_2; Obst_3; Antenna]';
ns     = (char('C','R1','R2', 'R3','C1'))'; % Names of objects
sf     = 'C-R1-R2-R3-C1';                   % Omega minus the 3 objects (holes)

[mesh,model] = fun_get_Mesh(Domain,sf,ns,H_max,H_min); % Assemble the geometry and build the mesh
% triplot(mesh.T); % Have a look on the mesh
% pdegplot(model,'EdgeLabels','on'); 
% --> Useful to check the edge labels to apply boundary conditions
% --> Another mesh file (for instance from GMSH) can be loaded, to skip all the code
%     until now. Just needs to set the good PET format.


%% ---------- Set the boundary conditions on the expected edges
mesh.e_Dir = [16 : 19]; % Labels of edges supporting inhom Dir cond (usually emitter)
mesh.e_ABC = [1 : 15];  % Labels of transparent edges
% --> Use pdegplot above to check edges labels.
% --> Here: apply 5 on Antenna (edges 12 to 15), hom Dir cond on facing obstacles (edges 2,3), 
%     ABC everywhere else.
% --> By default the non specified edges support hom Dir cond.
figure;
pdegplot(model, 'EdgeLabels', 'on');
title('Edge Labels for ABC');