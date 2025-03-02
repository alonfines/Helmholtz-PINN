# Helmholtz-PINN
readme.md‏
Physics-Informed Neural Network (PINN) for 2D Helmholtz Equation:

This project focuses on solving the 2D Helmholtz equation using PINN 

1. Problem Definition

We solve the 2D homogeneous Helmholtz equation:

\Deltas U + k^2 U = 0

on a defined computational domain, using a FEM-based approach. The Dirichlet boundary conditions and absorbing boundary conditions (ABC) are applied to model wave propagation scenarios.

2. Workflow

Step 1: Geometry Setup and FEM Solver (MATLAB)

The geometry of the problem is defined using MATLAB FEM scripts.

The main solver is based on the following MATLAB repository:

David Gasperini (2025), "FEM solver for 2D Helmholtz equation", https://www.mathworks.com/matlabcentral/fileexchange/91695-fem-solver-for-2d-helmholtz-equation.

Modifications:

The script Diffraction2.m was created to model wave propagation through a slit.

The output is stored in grid_data.mat, which contains both real and imaginary parts of the wave field.

Step 2: Data Processing (Python)

The Notebook.ipynb processes grid_data.mat, converting it into CSV format.

It extracts:

Either real or imaginary values of u from the solution into a csv file.

Boundary conditions at z=0, stored in bc_1.csv.

Both files should be saved in the baundary_conditions dir in the curriculum or seq2seq dir. 
