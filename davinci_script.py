import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from davinci_utils import prepare_dataset, normalize2
from torch.utils.data import DataLoader
import davinci_model
import sys
# sys.path.append('../Trajectory_simulation/lib')
import anomalous
import random

# Load LPTEM trajectories
lptem_all_shuffled_trainset = np.load('lptem_all_normalized2.npy')[0:28000]
print('Number of experimental trajectories: ', lptem_all_shuffled_trainset.shape[0])

#Generate synthetic trajectories
num_traj = 40000 - lptem_all_shuffled_trainset.shape[0]
num_brownian = int(0.2*(num_traj))
print('Number of simulated trajectories: ',num_traj)

num_hybrid = num_traj - num_brownian
print('Number of simulated brownian trajectories: ',num_brownian)
print('Number of simulated hybrid FBM/CTRW trajectories: ',num_hybrid)

timesteps = 200
np.random.seed(43)

train_dataset=np.empty([num_brownian,timesteps])

for i in range(num_brownian):
    x1 = np.empty([timesteps])
    train_dataset[i] = normalize2(anomalous.Brownian(N=timesteps-1,T=timesteps-1,delta=1)[0])

for i in range(num_hybrid):
    alpha = np.random.uniform(low=0.1, high=0.6)
    x1,_,_ = anomalous.CTRW(n=timesteps,alpha=alpha)
    while np.sum(x1**2)==0:
        x1,_,_ = anomalous.CTRW(n=timesteps,alpha=alpha)
    x1 = np.asarray(x1)
    x1 = normalize2(x1)

    x2 = np.empty([timesteps])
    alpha = np.random.uniform(low=0.1, high=0.6)
    x2,_,_,_ = anomalous.fbm_diffusion(n=timesteps,T=timesteps,H=alpha/2)
    x2 = np.asarray(x2)
    x2 = normalize2(x2)
    
    k = np.random.uniform(low=0,high=0.5)
    x3 = (1-k)*x1 + k*x2
    x3 = np.asarray(x3)
    x3 = normalize2(x3)

    train_dataset = np.concatenate((train_dataset,x3),axis=0)
    
train_dataset = np.concatenate([train_dataset,lptem_all_shuffled_trainset],axis=0)

trainset = prepare_dataset(train_dataset,shuffle=True,norm=True)
print('Trainset size: ',trainset.shape[0])

epochs = 1500  # Training epochs
batch_size = 1000 # Batch size
learning_rate = 3e-4   # Learning rate
weight_decay = 1e-5     # Weight decay


random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

dataloader= DataLoader(trainset, batch_size=batch_size,shuffle=True)

#Initialize the model
gpu = True 
device = torch.device('cuda:3' if gpu and torch.cuda.is_available() else 'cpu')   # Select GPU if available
model = davinci_model.TransformerVAE(device=device).to(device)
optimizer = torch.optim.Adam(model.parameters(),weight_decay=weight_decay,lr=learning_rate)     # Choose optimizer (Adam)

# Generate validation set
N_valset = 2000
timesteps = 200
np.random.seed(42)

validation_dataset=[]

for i in range(N_valset):
    x1 = np.empty([timesteps])
    alpha = np.random.uniform(low=0.1, high=0.6)
    x1,_,_ = anomalous.CTRW(n=timesteps,alpha=alpha)
    while np.sum(x1**2)==0:
        x1,_,_ = anomalous.CTRW(n=timesteps,alpha=alpha)

    x2 = np.empty([timesteps])
    alpha = np.random.uniform(low=0.1, high=0.6)
    x2,_,_,_ = anomalous.fbm_diffusion(n=timesteps,T=timesteps,H=alpha/2)

    k = np.random.uniform(low=0.1,high=0.5)
    x3 = (1-k)*x1 + k*x2

    validation_dataset.append(x3)

validation_dataset = np.asarray(validation_dataset)
validation_dataset = validation_dataset.reshape([validation_dataset.shape[0],validation_dataset.shape[-1]])
validationset = prepare_dataset(validation_dataset)
validation_loader = DataLoader(validationset, batch_size=batch_size)


#Train the model
model.train_davinci(epochs, dataloader, optimizer, device, val_loader=validation_loader)

#Save the model
parameters = {
    'batch size': batch_size,   
    'learning rate': learning_rate,
    'weight decay': weight_decay,
    'optimizer': 'Adam',
    'optimizer state dict': optimizer.state_dict(),
    'trajectory types': ['Brownian, CTRW, FBM, LPTEM'],
    'training dataset size': train_dataset.shape[0],
    }

model_name = 'DavinciModel2'
save_name = model_name

model.save_model(parameters, save_name)