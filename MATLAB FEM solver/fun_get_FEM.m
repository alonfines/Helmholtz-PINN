function[t_K,t_M,t_bM,I,J,I_b,J_b] = fun_get_FEM(mesh,id_e_ABC)
% Computes the area/length of all 2D/1D elements and the gradient of all associated
% shape functions. Then computes the mass and stiffness elementary matrices in fully
% vectorized way.


%% ---------- Elementary matrices of K and M
Node_1x = mesh.p(mesh.t(:,1),1).'; % Extract nodes numbers of the 3 vertices of each triangle 
Node_1y = mesh.p(mesh.t(:,1),2).'; 
Node_2x = mesh.p(mesh.t(:,2),1).'; 
Node_2y = mesh.p(mesh.t(:,2),2).'; 
Node_3x = mesh.p(mesh.t(:,3),1).'; 
Node_3y = mesh.p(mesh.t(:,3),2).';  

det = Node_1x .* Node_2y + Node_2x .* Node_3y + Node_3x .* Node_1y - ...
    Node_1x .* Node_3y - Node_2x .* Node_1y - Node_3x .* Node_2y; % Explicit formulation of the determinant of [1 Node_i_x Node_i_y], i = 1 : 3 (to evaluate the area)
det_Inv = 1./det;

Shape_temp1 = Node_2x .* Node_3y - Node_3x .* Node_2y; % Explicit formulation of the slopes of the shape functions (P1) associated to the nodes of the triangles
Shape_temp2 = -(Node_3y - Node_2y);
Shape_temp3 = Node_3x - Node_2x;
Shape_temp4 = -(Node_1x .* Node_3y - Node_3x .* Node_1y);
Shape_temp5 = Node_3y - Node_1y;
Shape_temp6 = -(Node_3x - Node_1x);
Shape_temp7 = Node_1x .* Node_2y - Node_2x .* Node_1y;
Shape_temp8 = -(Node_2y - Node_1y);
Shape_temp9 = Node_2x - Node_1x;

Shape_temp = [Shape_temp1; Shape_temp2; Shape_temp3; Shape_temp4; Shape_temp5; ...
    Shape_temp6; Shape_temp7; Shape_temp8; Shape_temp9]; % Table of the shape functions slopes (format 9 x nb_t)

Shape_mat  = Shape_temp .* det_Inv;

Grad = Shape_mat([2 3 5 6 8 9],:); % Gradients of the shape function (2 x 3 x nb_t reshaped in 6 x nb_t)
Area = 1/2 * abs(det);             % Areas of the triangles

t_K = bsxfun(@times,fun_get_LocProd(Grad),Area); % Computes the elementary stiffness 
t_M  = Area/12 .* [2; 1; 1; 1; 2; 1; 1; 1; 2];   % Explicit calculation of the mass elementary matrix (1 x 9 format)
% --> For unit reference element: Me_ref = 1/12 * [2 1 1
%                                                  1 2 1
%                                                  1 1 2]


%% ---------- Elementary matrices of bM
e_Vec = mesh.p(mesh.e(id_e_ABC,1),:) - mesh.p(mesh.e(id_e_ABC,2),:);
e_L   = sqrt(sum(e_Vec.^2,2));  % Lengths of transparent 1D elements
t_bM  = e_L'/6 .* [2; 1; 1; 2]; % Resolution of 1D integral


%% ---------- Global indices
I = mesh.t(:,[1 2 3 1 2 3 1 2 3]).'; % Global indices, rows
J = mesh.t(:,[1 1 1 2 2 2 3 3 3]).'; % Global indices, columns

I_b = mesh.e(id_e_ABC(:),[1 2 1 2]).'; % Same for edge elements
J_b = mesh.e(id_e_ABC(:),[1 1 2 2]).';