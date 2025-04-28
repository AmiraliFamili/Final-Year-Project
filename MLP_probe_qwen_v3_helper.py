from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class HiddenStatesDataset(Dataset):
    def __init__(self, hidden_states, labels):
        self.hidden_states = hidden_states
        self.labels = labels.astype(np.int64)
        
    def __len__(self):
        return len(self.hidden_states)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.hidden_states[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

class EnhancedProbe(nn.Module):
    """Enhanced probe with multiple hidden layers and dropout."""
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.3):
        super(EnhancedProbe, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.GELU())
            #layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
