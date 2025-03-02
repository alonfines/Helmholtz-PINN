import os
import torch
import wandb
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from model_curriculum import HelmholtzPINN, set_seed, SEED
from dataset_curriculum import HelmholtzDataset, custom_collate
from utils_curriculum import config

# Add argument parser
set_seed()
parser = argparse.ArgumentParser(description='Train Helmholtz PINN')
parser.add_argument('--k_squared', type=int, required=False, default=1,
                   help='Sequence number for training (use "full" for full domain or an integer)')
args = parser.parse_args()

# Convert seq_num to proper type
k_squared = int(args.k_squared)

config["parameters"]["k_squared"] = k_squared

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set wandb to offline mode (skip visualization)
os.environ["WANDB_MODE"] = "offline"
wandb.init(project="helmholtz-pinn")

k_squared = config["parameters"]["k_squared"]
batch_size = config["parameters"]["batch_size"]
num_samples = config["parameters"]["n_points"]
dt = config["parameters"]["dt"]
seq_num = config["parameters"]["seq_num"]

# Log hyperparameters to wandb
wandb.config.update({
    "k_squared": k_squared,
    "n_points": num_samples
})

x_train = torch.rand((num_samples, 2))  # Generate random points in the domain [0, 1] x [0, 1]
x_train[:, 0] = x_train[:, 0]*config["parameters"]["L_x"]
x_train[:, 1] = x_train[:, 1]*config["parameters"]["z"]

train_dataset = HelmholtzDataset(x_train)
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,  
    num_workers=8,  
    pin_memory=True,
    persistent_workers=True, 
    collate_fn=custom_collate
)

model = HelmholtzPINN()

current_dir = os.path.dirname(__file__)
checkpoint_dir = config["paths"]["checkpoint_dir"]
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_path = os.path.join(current_dir, checkpoint_dir)+ f'helmholtz_checkpoint{k_squared-1}.ckpt' if k_squared > 1 else None

checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename=f'helmholtz_checkpoint{k_squared}',
    save_top_k=1,
    save_last=False,
    mode='min',
    monitor='watch_loss'
)

early_stop_callback = EarlyStopping(
    monitor='watch_loss',
    patience=200,
    stopping_threshold = 2.5e-3,
    verbose=True,
    mode='min'
)

wandb_logger = WandbLogger(project="helmholtz-pinn")

trainer = pl.Trainer(
    max_epochs=config["parameters"]["epochs"],
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=wandb_logger,
    log_every_n_steps=1,
    enable_progress_bar=True,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    detect_anomaly=True,
)

wandb_logger.watch(model, log="all")

# Load previous checkpoint if it exists
if checkpoint_path and os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Load only model weights, not boundary conditions
    weights = {k: v for k, v in checkpoint['state_dict'].items() 
              if not any(k.startswith(prefix) for prefix in 
                        ['x_boundary', 'z_rbc', 'u_bc', 'xz_boundary'])}
    
    # Load filtered weights into model
    model.load_state_dict(weights, strict=False)

# Start training
trainer.fit(model, train_loader)

# Close wandb run
wandb.finish()