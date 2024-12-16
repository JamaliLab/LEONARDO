import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from LEONARDO_utils import prepare_dataset
from torch.utils.data import DataLoader
import LEONARDO_model
import sys
sys.path.append('../../../Trajectory_simulation/lib')
import anomalous, goldenrod, models_theory
from models_theory import models_theory
import random

gpu_number = 3
# Load LPTEM trajectories
trainset = np.load('lptem_train.npy')[:,:,0:2]
trainset_torch = prepare_dataset(trainset,shuffle=True,norm=True)
print(trainset_torch.shape)

valset = np.load('lptem_val.npy')[:,:,0:2]
valset_torch = prepare_dataset(valset,shuffle=True,norm=True)


epochs = 200  # Training epochs
batch_size = 1000 # Batch size
learning_rate = 3e-4   # Learning rate
weight_decay = 1e-5     # Weight decay


random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

dataloader= DataLoader(trainset_torch, batch_size=batch_size, num_workers=4,shuffle=True)
validation_loader = DataLoader(valset_torch, batch_size=batch_size, num_workers=4, shuffle=True)



#Initialize the model
gpu = True 
device = torch.device(f'cuda:{gpu_number}' if gpu and torch.cuda.is_available() else 'cpu')   # Select GPU if available
model = LEONARDO_model.TransformerVAE(device=device).to(device)
optimizer = torch.optim.Adam(model.parameters(),weight_decay=weight_decay,lr=learning_rate)     # Choose optimizer (Adam)


#Train the model
model.train_Leonardo(epochs, dataloader, optimizer, device, val_loader=validation_loader)

#Save the model


parameters = {
    'Batch Size': batch_size,   
    'Learning Rate': learning_rate,
    'Weight Decay': weight_decay,
    'Optimizer': 'Adam',
    'optimizer state dict': optimizer.state_dict(),
    'Trajectory Types': 'LPTEM',
    'Training Dataset Size': trainset_torch.shape[0],
    'Validation Dataset Size': valset_torch.shape[0],
    'Random Seed': random_seed,
    'GPU number': gpu_number,
    'Trained on dx or x?': 'x',
    'Normalization type': '0 to 1',
    'Notes': ""
    }
model_name = 'models_2d/Leonardo'
save_name = model_name

model.save_model(parameters, save_name)
print(f'Model name: {model_name}')

