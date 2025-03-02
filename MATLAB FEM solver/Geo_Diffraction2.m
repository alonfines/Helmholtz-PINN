% SET GEOMETRY & MESH 

%% ---------- Define the geometry
% --> Quadrangle format 1 x 10: [3 4 x_UpLeft x_UpRight x_DownRight x_DownLeft y_UpLeft y_UpRight y_DownRight y_DownLeft]
% --> Circle format 1 x 4: [1 x_c y_c radius]

Omega   = [3 4 0 1 1 0 3 3 0 0];  % Square domain                   % Unit square domain
Obst_1  = [3 4 -0.01 0.45 0.45 -0.01 1.01 1.01 1 1]; % Obstacle 1: thin rectangle
Obst_3  = [3 4 0.55 1 1 0.55 1.01 1.01 1 1]; % Obstacle 3: thin rectangle
Antenna = [1 0.5 2.05 0.005 zeros(1,6)];                  % Small circle for source term


%% ---------- Define the computational domainand build the mesh
Domain = [Omega; Obst_1; Obst_3; Antenna]';
ns     = (char('C','R1','R3','C1'))'; % Names of objects
sf     = 'C-R1-R3-C1';                   % Omega minus the 3 objects (holes)

[mesh,model] = fun_get_Mesh(Domain,sf,ns,H_max,H_min); % Assemble the geometry and build the mesh
% triplot(mesh.T); % Have a look on the mesh
% pdegplot(model,'EdgeLabels','on'); 
% --> Useful to check the edge labels to apply boundary conditions
% --> Another mesh file (for instance from GMSH) can be loaded, to skip all the code
%     until now. Just needs to set the good PET format.


%% ---------- Set the boundary conditions on the expected edges
mesh.e_Dir = [12 : 15]; % Labels of edges supporting inhom Dir cond (usually emitter)
mesh.e_ABC = [1 : 11];  % Labels of transparent edges
% --> Use pdegplot above to check edges labels.
% --> Here: apply 5 on Antenna (edges 12 to 15), hom Dir cond on facing obstacles (edges 2,3), 
%     ABC everywhere else.
% --> By default the non specified edges support hom Dir cond.c
% figure;
% pdegplot(model, 'EdgeLabels', 'on');
% title('Edge Labels for ABC');