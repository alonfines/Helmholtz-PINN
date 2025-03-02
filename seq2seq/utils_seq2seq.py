import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.io
from scipy.interpolate import griddata
import yaml
import pandas as pd
import random
import argparse

# Global seed constant
SEED = 42

def set_seed(seed=SEED):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Helmholtz PINN')
    parser.add_argument('--seq_num', type=int, required=False, default=0,
                       help='Sequence number for training/testing (use "full" for full domain or an integer)')
    args = parser.parse_args()
    # Convert seq_num to proper type and update config
    config["parameters"]["seq_num"] = 'full' if args.seq_num == 0 else int(args.seq_num)
    return args

# Load config
config_path = os.path.join(os.path.dirname(__file__), 'config_seq2seq.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

def create_test_grid(seq_num, dt, n_points=1000):
    """Create evaluation grid"""
    x = np.linspace(0, 6, n_points)
    if seq_num == 0:
        y = np.linspace(0, 3, n_points)
    else:   
        y = np.linspace(0, seq_num*dt, 100)
    xx, yy = np.meshgrid(x, y)
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]

def load_fem_data(config, xx, yy):
    """Load and interpolate FEM solution"""
    current_dir = os.path.dirname(__file__)
    data_rel_path = config["paths"]["grid_data"]
    data_path = os.path.join(current_dir, data_rel_path)
    data = pd.read_csv(data_path)
    x_values = data['x'].values
    z_values = data['z'].values
    u_values = data['u'].values

    fem = griddata((x_values, z_values), u_values, (xx, yy), method='cubic')
    
    return fem 

def plot_solutions(xx, yy, pinn_sol, fem, config, plots_dir):
    """Create and save comparison plots for real and imaginary components"""
    
    # Create figure with 2 rows (real/imag) and 2 columns (PINN/FEM)
    fig, axes = plt.subplots(1, 2, figsize=(15, 12))
    
    # Determine common color scale for real components
    vmin_real = min(pinn_sol.min(), fem.min())
    vmax_real = max(pinn_sol.max(), fem.max())
    
    
    # Plot components
    c1 = axes[0].contourf(xx, yy, pinn_sol, levels=50, cmap='jet', 
                           vmin=vmin_real, vmax=vmax_real)
    axes[0].set_title(f'PINN amplitude (k^2={config["parameters"]["k_squared"]})')
    
    c2 = axes[1].contourf(xx, yy, fem, levels=50, cmap='jet',
                           vmin=vmin_real, vmax=vmax_real)
    axes[1].set_title('FEM amplitude')
    

    # Add labels and colorbars
    for ax in axes.flat:
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
    
    # Add colorbars
    fig.colorbar(c1, ax=axes[0], label="u Value")
    fig.colorbar(c2, ax=axes[1], label="u Value")

    
    # Adjust layout
    plt.tight_layout()
    
    # Calculate errors for filename
    real_mse = np.sqrt(np.mean((pinn_sol - fem)**2))
    total_mse = np.sqrt(real_mse**2)
    
    # Save plot
    filename = (f'helmholtz_real_imag_comparison_k^2={config["parameters"]["k_squared"]}'
               f'_seq_num={config["parameters"]["seq_num"]}'
               f'_total_mse={total_mse:.4f}.png')
    plot_filename = os.path.join(plots_dir, filename)
    
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_filename

def calculate_l2_error(pinn_sol, fem_sol):
    """Calculate L2 error between PINN and FEM solution amplitudes"""
    pinn_amplitude = np.sqrt(pinn_sol**2 + pinn_sol**2)
    fem_amplitude = np.sqrt(fem_sol**2 + fem_sol**2)
    return np.sqrt(np.mean((pinn_amplitude - fem_amplitude)**2))

def load_boundary_conditions(seq_num):
    """Load boundary conditions from CSV file"""
    current_dir = os.path.dirname(__file__)
    bc_dir = config["paths"]["bc_dir"]
    if seq_num == 0:
            bc_path = os.path.join(current_dir,bc_dir)+'/bc1.csv'
  # Get path from config
    else:
        bc_path = os.path.join(current_dir,bc_dir)+f'/bc{seq_num}.csv'  # Get path from config
    data = pd.read_csv(bc_path)
    
    # Convert to torch tensors
    boundary_points = torch.tensor(data[['x', 'z']].values, dtype=torch.float32)
    u_bc = torch.tensor(data['u'].values, dtype=torch.float32)
    
    return boundary_points, u_bc