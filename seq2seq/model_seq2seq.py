import torch
import torch.nn as nn
import pytorch_lightning as pl
import yaml
from utils_seq2seq import load_boundary_conditions, config, SEED, set_seed, get_args
from torch.autograd import grad
import numpy as np
from typing import Dict, Any
import os

# Load config
config_path = os.path.join(os.path.dirname(__file__), 'config_seq2seq.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Override seq_num if provided via command line
args = get_args()
if args.seq_num is not None:
    config["parameters"]["seq_num"] = args.seq_num

seq_num = config["parameters"]["seq_num"]
k_squared = config["parameters"]["k_squared"]


class HelmholtzPINN(pl.LightningModule):
    def __init__(self, bc_create=False):
        super().__init__()
        self.save_hyperparameters()
        self.k_squared = float(k_squared)

        
        # Load boundary conditions based on bc_create flag
        if bc_create:
            boundary_points, u_bc = load_boundary_conditions(seq_num-1)  # Load previous sequence
        else:
            if seq_num == 'full':
                boundary_points, u_bc= load_boundary_conditions(1) # Load first boundary conditions
            else:
                boundary_points, u_bc = load_boundary_conditions(seq_num)    # Load current boundary conditions
        
        # Register boundary conditions as buffers
        self.register_buffer('xz_boundary', boundary_points)
        self.register_buffer('u_bc', u_bc)

        self.model = nn.Sequential(
                nn.Linear(2, 50),  # Increase width
                nn.Tanh(),
                nn.Linear(50, 50),
                nn.Tanh(),
                nn.Linear(50, 50),
                nn.Tanh(),
                nn.Linear(50, 1),  # Add more layers
            
)
        
        
    def forward(self, x):
        return self.model(x)

    def helmholtz_residual(self, x):
        """Modified Helmholtz residual calculation for real component"""
        x.requires_grad_(True)
        output = self(x)  

        u_image = output  
        grad_u_image = grad(u_image.sum(), x, create_graph=True)[0]
        u_image_x, u_image_y = grad_u_image[:, 0], grad_u_image[:, 1]
        u_image_xx = grad(u_image_x.sum(), x, create_graph=True)[0][:, 0]
        u_image_yy = grad(u_image_y.sum(), x, create_graph=True)[0][:, 1]
        image_residual = u_image_xx + u_image_yy + self.k_squared * u_image.squeeze()
        
        return image_residual 

    def boundary_loss(self):
        """Calculate boundary loss with proper gradient tracking"""
        # Use registered buffers and ensure gradient tracking
        xz_boundary = self.xz_boundary.detach().clone().requires_grad_(True)  # Correct way to enable gradients
        u_bc = self.u_bc
        
        # Compute predictions
        output = self(xz_boundary)
        pred_ibc = output
        
        l2_ibc_loss = torch.mean((pred_ibc.squeeze() - u_bc) ** 2)
        
        # Combine both losses for better boundary fitting
        return l2_ibc_loss

    def configure_optimizers(self):
        """Configure Adam optimizer with learning rate scheduler"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(config["parameters"]["learning_rate"]),  # Initial learning rate
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor = float(config["parameters"]["factor"]),
            patience = float(config["parameters"]["patience"]),
            min_lr = float(config["parameters"]["min_lr"]),
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "watch_loss"
            }
        }

    def training_step(self, batch, batch_idx):
        pde_loss = torch.mean(self.helmholtz_residual(batch) ** 2)
        bc_loss = self.boundary_loss()

        # Check for invalid values
        if torch.isnan(pde_loss) or torch.isnan(bc_loss):
            self.log('nan_detected', 1.0)
            return None

        # Define threshold
        epsilon = float(config["parameters"]["error"])

        # Compute relative deviation (avoid divide-by-zero)
        rel_pde = torch.abs(pde_loss - epsilon) / (pde_loss +bc_loss+ epsilon + 1e-8)
        rel_bc = torch.abs(bc_loss - epsilon) / (pde_loss+bc_loss + epsilon + 1e-8)

        # Clamp values to prevent overflow issues
        rel_pde = torch.clamp(rel_pde, max=10.0)  # Prevent large exponents
        rel_bc = torch.clamp(rel_bc, max=10.0)

        # Compute adaptive lambda weights **without in-place modification**
        lambda1 = torch.exp(rel_pde.clone().detach())  # Clone and detach before exp
        lambda2 = torch.exp(rel_bc.clone().detach())

        
        # Handle cases where one loss is much smaller than epsilon
        if bc_loss >= epsilon:
            lambda1 = torch.tensor(0.1, device=pde_loss.device, dtype=pde_loss.dtype)
            lambda2 = torch.tensor(9, device=pde_loss.device, dtype=pde_loss.dtype)
        elif bc_loss < epsilon and pde_loss >= epsilon:
            lambda1 = torch.tensor(0.8, device=pde_loss.device, dtype=pde_loss.dtype)
            lambda2 = torch.tensor(0.2, device=pde_loss.device, dtype=pde_loss.dtype)
        elif pde_loss < epsilon and bc_loss < epsilon:
            lambda1 = torch.tensor(0.5, device=pde_loss.device, dtype=pde_loss.dtype)
            lambda2 = torch.tensor(0.5, device=pde_loss.device, dtype=pde_loss.dtype)

        # Normalize weights (avoid in-place operations)
        lambda_sum = lambda1 + lambda2 + 1e-8  # Avoid division by zero
        lambda1 = lambda1 / lambda_sum
        lambda2 = lambda2 / lambda_sum

        # Modify total loss to include zero penalty
        total_loss = lambda1 * pde_loss + lambda2 * bc_loss
        self.log_dict({
            'watch_loss': pde_loss + bc_loss,
            'train_loss': total_loss,
            'pde_loss': pde_loss,
            'bc_loss': bc_loss,
            'lambda1': lambda1,
            'lambda2': lambda2,
        }, prog_bar=True)

        return total_loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Custom optimizer step for LBFGS"""
        optimizer.step(optimizer_closure)

    def test_step(self, batch, batch_idx):
        """Evaluation step for testing and validation"""
        # Compute losses with gradient tracking for PDE residuals
        with torch.enable_grad():
            x = batch[:, 0].clone().requires_grad_(True).view(-1, 1)
            y = batch[:, 1].clone().requires_grad_(True).view(-1, 1)
            grid_points = torch.cat([x, y], dim=1)
            
            pde_loss = torch.mean(self.helmholtz_residual(grid_points)**2)
            bc_loss = self.boundary_loss()
        
        # Get predictions for the current batch
        predictions = self(batch)
        
        # Log metrics
        self.log_dict({
            'test_pde_loss': pde_loss,
            'test_bc_loss': bc_loss,
        }, prog_bar=True)
        
        return {
            'predictions': predictions,
            'pde_loss': pde_loss,
            'bc_loss': bc_loss,
        }

