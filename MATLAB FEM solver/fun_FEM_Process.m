function[U] = fun_FEM_Process(k,Amp_source,mesh)
% FEM processing: P1 matrices assembly and resolution. The assembly of the
% mass, edge mass and the stiffness are fully vectorized, according to [1]. A
% "pedagogic" version of this assembly is provided in comment at the end of
% this file.


%% ---------------------------------------------------------------
%% | Find id's of physical regions to handle boundary conditions |
%% ---------------------------------------------------------------
mesh.nb_t   = size(mesh.t,1); % Number of elements
mesh.nb_p   = size(mesh.p,1); % Number of nodes

id_p_Dir  = []; % Initialize id arrays

for i = 1 : length(mesh.e_Dir) % Run the inhom Dir conds
    id_e_Dir_loc = find(ismember(mesh.e(:,3),mesh.e_Dir(i))); % Id's of edges (1D elements) of bound e_Dir(i)
    id_p_Dir_loc = unique(mesh.e(id_e_Dir_loc,1 : 2));        % Id's of corresponding nodes
    id_p_Dir     = unique([id_p_Dir(:); id_p_Dir_loc(:)]);    % Build the id array in good order
end

id_Tot      = unique(mesh.e(:,3));                       % Labels of the different physical regions
id_e_ABC    = find(ismember(mesh.e(:,3),mesh.e_ABC));    % Id's of edges of transparent bounds
id_e_HomDir = find(ismember(mesh.e(:,3),setdiff(id_Tot,[mesh.e_Dir,mesh.e_ABC]))); % Id's of edges with hom Dir cond (unspecified remaining bounds)
id_p_HomDir = unique(mesh.e(id_e_HomDir,1 : 2));         % Id's of corresponding nodes
id_p_TotDir = [id_p_Dir; setdiff(id_p_HomDir,id_p_Dir)]; % Full array of Dir nodes
id_DoF      = setdiff((1 : mesh.nb_p).',id_p_TotDir);    % Id's of free nodes (ABC nodes + interior nodes)

bound_Dir   = [Amp_source * ones(length(id_p_Dir),1); ...
               zeros(length(setdiff(id_p_TotDir,id_p_Dir)),1)]; % Take in account hom Dir cond in boundary values


%% ---------------------------
%% | FEM vectorized assembly |
%% ---------------------------
% --> The mass M, stiffness K and edge mass bM are assembled as follows:
%     - compute the 3 x 3 elementary matrices of all elements using vectorized operations
%     - reshape each elementary matrix into 9 x 1
%     - store each of them in corresponding full tables (size 9 x number of elements) t_M, t_K and t_bM.
%       This allows to optimize the storage since we only store useful data, and do not rewrite 
%       in a sparse matrix at each iteration [1]
%     - compute the resulting Helmholtz system table t_H (without ABC): 
%       t_H = k^2 * t_M - t_K
%     - build the suitable global indices I, J and I_b, J_b in order to
%       spread t_H and t_bM into the matrices H and bM
%     - apply the ABC: A = H + 1i*k * bM

[t_K,t_M,t_bM,I,J,I_b,J_b] = fun_get_FEM(mesh,id_e_ABC);

t_H = k^2 * t_M - t_K;                          % Table with elementary matrices of Helmholtz system
H   = sparse(I,J,t_H,mesh.nb_p,mesh.nb_p);      % Assemble sparse FEM matrix
bM  = sparse(I_b,J_b,t_bM,mesh.nb_p,mesh.nb_p); % Assemble sparse FEM matrix for transparent bounds
% --> Put t_H = -t_K and A = H to treat Laplace equation

A = H + 1i*k  * bM;       % Assemble global coefficients matrix
A_DoF = A(id_DoF,id_DoF); % Extract the DoF's for the resolution (Dir nodes sent to right hand side)

b = -A(id_DoF,id_p_TotDir) * bound_Dir; % Right hand side, enforcing the Dir conds


%% --------------
%% | Resolution |
%% --------------
% --> The direct solver \ is used. For this pde, it leads to undefinite
%     symmetric matrix A, and hence to LU factorization.

U_vect = A_DoF \ b;          % Solve over the DoF's
U      = zeros(mesh.nb_p,1); % Initialize the full solution vector
U(id_p_TotDir) = bound_Dir;  % Fill the solution with Dir nodes (Antenna(s) + reflecting parts)
U(id_DoF)      = U_vect;     % Fill with DoF's 


%% --------------------------
%% | Simple way to assemble |
%% --------------------------
% t_M  = zeros(9,mesh.nb_t); % Initialize tables (i-th column = reshaped elementary matrix of triangle i)
% t_K  = zeros(9,mesh.nb_t);
% 
% for i = 1 : mesh.nb_t  % Loop over the triangles
% % --- Characterization of triangular element i
%     nodes_t = mesh.t(i,:);         % Id's of the 3 nodes of triangle i
%     Pe      = [ones(3,1), mesh.p(nodes_t,1:2)];  % 3x3 matrix with lines (1 Xcorner Ycorner)
%     CoefT   = inv(Pe);             % Columns of CoefT are coefs a,b,c in shape function phi = a+bx+cy 
%     Area    = abs(det(Pe))/2;
% % --- Elementary stiffness of element i 
%     grad = CoefT(2:3,:);           % P1 elements: grad = the coefs
%     Ke   = Area * (Grad.' * Grad); % Elementary stiffness
%     Ke   = reshape(Ke,9,1);        % Reshape in 9x1 to load in tK
% % --- Elementary mass of element i
%     Me_ref = [2 1 1; 1 2 1; 1 1 2] % Elementary mass of reference element
%     Me     = Area/12 * Me_ref      % Elementary mass of element i
%     Me = reshape(Me,9,1);          % Reshape in 9x1 to load in tM
% % --- Load the elementary matrices on the tables
%     tM(:,i) = Me; 
%     tK(:,i) = Ke; 
% end