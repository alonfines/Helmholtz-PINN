import torch
from torch.utils.data import Dataset

# ---------------------------
# Data Preparation
# ---------------------------
class HelmholtzDataset(Dataset):
    def __init__(self, points):
        self.points = points
    
    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, idx):
        return self.points[idx]

def custom_collate(batch):
    stacked = torch.stack(batch)
    return stacked