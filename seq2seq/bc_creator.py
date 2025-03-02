from scipy.io import loadmat
import torch
import numpy as np
import os
import pandas as pd
from utils_seq2seq import config, set_seed, get_args
from model_seq2seq import HelmholtzPINN

def save_boundary_conditions(boundary_points, u_bc, seq_num):
    """Save boundary conditions to CSV"""
    # Combine numpy operations instead of individual conversions
    data = pd.DataFrame(
        np.column_stack([
            boundary_points.numpy(),
            u_bc.numpy().reshape(-1, 1),
        ]),
        columns=['x', 'z', 'u']
    )
    cuurent_dir = os.path.dirname(__file__)
    bc_dir = config["paths"]["bc_dir"]
    bc_path = os.path.join(cuurent_dir, bc_dir)+f"/bc{seq_num}.csv"
    data.to_csv(bc_path, index=False)

def create_bc(seq_num):
    """Create boundary conditions and save them"""
    # Load data once at the start
    
    if seq_num == 1:
        return None, None

    elif seq_num == 'full':
        return None, None
    else:
        # Load model and predict
        model = HelmholtzPINN(bc_create=True)
        # Add device handling for model loading
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        current_dir = os.path.dirname(__file__)
        checkpoint_dir = config["paths"]["checkpoint_dir"]
        checkpoint_path = os.path.join(current_dir, checkpoint_dir)+f"/helmholtz_checkpoint{seq_num-1}.ckpt"
        checkpoint = torch.load(checkpoint_path, 
                              map_location=device)
        weights = {k: v for k, v in checkpoint['state_dict'].items() 
              if not any(k.startswith(prefix) for prefix in 
                        ['x_boundary', 'z_rbc', 'u_bc', 'xz_boundary'])}
    
    # Load filtered weights into model
        model.load_state_dict(weights, strict=False)
        model.to(device)
        model.eval()
        
        x_bc = torch.linspace(0, 6, 1000, dtype=torch.float32)
        y_bc = torch.ones_like(x_bc) * (seq_num-1) * config['parameters']['dt']
        # Stack points before prediction
        boundary_points = torch.stack([x_bc, y_bc], dim=1).to(device)
        with torch.no_grad():
            output = model(boundary_points).squeeze().cpu()
            z_ibc= output
    
    # Stack points after all calculations
    boundary_points = torch.stack([x_bc, y_bc], dim=1)
    
    # Save results
    save_boundary_conditions(boundary_points, z_ibc, seq_num)
    return boundary_points, z_ibc

if __name__ == "__main__":
    # Get command line arguments and update config
    args = get_args()
    seq_num = config["parameters"]["seq_num"]  # Now comes from parsed args
    
    # Enable CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and save boundary conditions
    boundary_points, u_bc = create_bc(seq_num)
    
    # # Print information
    # print(f"Boundary points shape: {boundary_points.shape}")
    # print(f"Z imag values shape: {z_ibc.shape}")
    # print(f"The seq_num is: {seq_num}")