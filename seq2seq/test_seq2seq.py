import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from model_seq2seq import HelmholtzPINN, SEED, set_seed
import scipy.io
from scipy.interpolate import griddata
from utils_seq2seq import config, create_test_grid, load_fem_data, plot_solutions, calculate_l2_error
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import argparse


def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description='Test Helmholtz PINN')
    parser.add_argument('--seq_num', type=str, required=True, default=0,
                       help='Sequence number for testing (use "full" for full domain or an integer)')
    args = parser.parse_args()

    # Update config with parsed seq_num
    config['parameters']['seq_num'] = 'full' if args.seq_num == 0 else int(args.seq_num)

    set_seed()  # Use default SEED=42
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plots_dir = os.path.join(os.path.dirname(__file__), config['paths']['plots_dir'])
    os.makedirs(plots_dir, exist_ok=True)
    
    # Model setup
    model = HelmholtzPINN().to(device)
    current_dir = os.path.dirname(__file__)
    checkpoint_dir = os.path.join(current_dir, config['paths']['checkpoint_dir'])

    checkpoint_path = checkpoint_dir + f"/helmholtz_checkpoint{config['parameters']['seq_num']}.ckpt"
    print(f"------------------------------Loading checkpoint from------------- \n{checkpoint_path}\n")
    checkpoint = torch.load(checkpoint_path, 
                              map_location=device)
    weights = {k: v for k, v in checkpoint['state_dict'].items() 
              if not any(k.startswith(prefix) for prefix in 
                        ['x_boundary', 'z_rbc', 'u_bc', 'xz_boundary'])}
    
    # Load filtered weights into model
    model.load_state_dict(weights, strict=False)
    model.to(device)
    model.eval()
    
    # Create evaluation grid
    xx, yy, grid_points = create_test_grid(
        config['parameters']['seq_num'],
        config['parameters']['dt']
    )
    
    # Create dataloader for testing
    test_data = torch.tensor(grid_points, dtype=torch.float32)
    test_loader = DataLoader(
        TensorDataset(test_data),
        batch_size=1024,
        shuffle=False
    )
    
    # Get predictions
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch[0].to(device)
            output = model.test_step(batch, 0)
            predictions.append(output['predictions'].cpu())
    
    predictions = torch.cat(predictions, 0).numpy()
    pinn_solution_real = predictions.reshape(xx.shape)

    # Load and interpolate FEM solution
    fem_solution_real = load_fem_data(config, xx, yy)
    
    # Plot and save results
    plot_filename = plot_solutions(
        xx, yy, 
        pinn_solution_real,
        fem_solution_real, 
        config, plots_dir
    )
    print(f"Plot saved as {plot_filename}")
    


if __name__ == "__main__":
    main()