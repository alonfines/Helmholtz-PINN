function[mesh,model] = fun_get_Mesh(Domain,sf,ns,H_max,H_min)
% Build the mesh using built-in Matlab functions. 
% The class model only serves to plot the edges labels to apply the 
% boundary conditions. The class mesh contains all the mesh stats.
% Generates the mesh from geometry Domain with structure sf and names ns.


model = createpde; % Set model geometry
geom  = decsg(Domain,sf,ns);
geometryFromEdges(model,geom);
generateMesh(model,'Hmax',H_max,'Hmin',H_min,'GeometricOrder','Linear'); % Hmin and Hmax give min and max element size (adapted mesh)

[p,e,t] = meshToPet(model.Mesh); % PET format: points, edges and triangles

mesh.t = t(1 : 3, :)'; % Load to class mesh
mesh.p = p';
mesh.e = e([1 2 5],:)';

mesh.T = triangulation(mesh.t(:,1 : 3),mesh.p); % Triangulation data from mesh