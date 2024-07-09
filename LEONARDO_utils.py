import numpy as np    
import torch
import sys
import pathlib
import pandas as pd
# sys.path.append('../Trajectory_simulation/lib/')
from pytorch_forecasting.utils import autocorrelation
import os
import random

gpu = True 
device = torch.device('cuda:3' if gpu and torch.cuda.is_available() else 'cpu')   # Select GPU if available

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

## Set Random Seeds
random_seed = 42
os.environ['PYTHONHASHSEED'] = str(random_seed)
torch.use_deterministic_algorithms(True)

np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize(trajs):
    """Normalizes trajectories between -1 and 1
    """
    if len(trajs.shape) == 1:
        trajs = np.reshape(trajs, (1, trajs.shape[0]))
    new_trajs = np.empty([trajs.shape[0],trajs.shape[1]])
    max = np.empty([trajs.shape[0],1])
    for i,traj in enumerate(trajs):
        traj[:] = traj[:] - traj[0]
        max[i] = np.max(np.absolute(traj))
        if max[i] == 0:
            max[i] = 1
        new_trajs[i] = (traj)/max[i]
    return new_trajs


def normalize2(trajs):
    """Normalizes trajectories between 0 and 1
    """
    if len(trajs.shape) == 1:
        trajs = np.reshape(trajs, (1, trajs.shape[0]))
    new_trajs = np.empty([trajs.shape[0],trajs.shape[1]])
    max_min = np.empty([trajs.shape[0],1])
    for i,traj in enumerate(trajs):
        max_min[i] = (np.max(traj)-np.min(traj))
        if max_min[i] == 0:
            max_min[i] = 1
        new_trajs[i] = (traj - np.min(traj))/max_min[i]
    return new_trajs


def prepare_dataset(dataset, return_labels=False, n_labels=0, norm=True, shuffle=True, to_tensor=True, test=False):
    """ Normalizes trajectories from dataset, shuffles them and converts them from np arrays to torch tensors 
    """
    if shuffle: np.random.shuffle(dataset)
    if test: n_labels=1
    labels = dataset[:,:n_labels]
    dataset = dataset[:,n_labels:]
    if norm: dataset = normalize(dataset)
    if to_tensor: dataset = torch.from_numpy(dataset).float()
    if return_labels: return dataset, labels
    return dataset


def kurtosis_dx(traj, norm=True):
    """ Calculates the kurtosis of displacements of given trajectories in numpy
    """
    traj_dx = traj[:,1:] - traj[:,0:-1] # Calculate trajectory dxs
    traj_dx_mean = np.mean(traj_dx, axis=1, keepdims=True) # Calculate the mean of each trajectory dx. Keep dimensions for broadcasting
    traj_dx_centered = traj_dx - traj_dx_mean # Center the trajectories by subtracting their means
    traj_dx_kurtosis_dist_num = traj_dx_centered**4 # Calculate the fourth power of the centered trajectory dxs
    traj_dx_kurtosis_dist_den = traj_dx_centered**2 # Calculate the second power of the centered trajectory dxs
    if norm:
        traj_dx_kurtosis = np.average(traj_dx_kurtosis_dist_num, axis=1) / np.average(traj_dx_kurtosis_dist_den, axis=1)**2 # Calculate kurtosis by normalizing with the square of variance of the trajectory
    else:
        traj_dx_kurtosis = np.average(traj_dx_kurtosis_dist_num, axis=1) # Calculate kurtosis without normalizing
    return traj_dx_kurtosis


def skewness_dx(traj, norm=True):
    """ Calculates the skewness of displacements of given trajectories in numpy
    """
    traj_dx = traj[:,1:] - traj[:,0:-1] # Calculate trajectory dxs
    traj_dx_mean = np.mean(traj_dx, axis=1, keepdims=True) # Calculate the mean of each trajectory dx. Keep dimensions for broadcasting
    traj_dx_centered = traj_dx - traj_dx_mean # Center the trajectories by subtracting their means
    traj_dx_skewness_dist_num = traj_dx_centered**3 # Calculate the third power of the centered trajectory dxs
    traj_dx_skewness_dist_den = traj_dx_centered**2 # Calculate the second power of the centered trajectory dxs
    if norm:
        traj_dx_skewness = np.average(traj_dx_skewness_dist_num, axis=1) / np.average(traj_dx_skewness_dist_den, axis=1)**1.5 # Calculate skewness by normalizing with the cube of standard deviation of the trajectory
    else:
        traj_dx_skewness = np.average(traj_dx_skewness_dist_num, axis=1) # Calculate skewness without normalizing
    return traj_dx_skewness


def variance_dx(traj):
    """ Calculates the variance of displacements of given trajectories in numpy
    """
    traj_dx = traj[:,1:] - traj[:,0:-1] # Calculate trajectory dxs
    traj_dx_mean = np.mean(traj_dx, axis=1, keepdims=True) # Calculate the mean of each trajectory dx. Keep dimensions for broadcasting
    traj_dx_centered = traj_dx - traj_dx_mean # Center the trajectories by subtracting their means
    traj_dx_variance = np.average(traj_dx_centered**2, axis=1) # Calculate the variance
    return traj_dx_variance


def mean_dx(traj):
    """ Calculates the skewness of displacements of given trajectories in numpy
    """
    traj_dx = traj[:,1:] - traj[:,0:-1] # Calculate trajectory dxs
    traj_dx_mean = np.mean(traj_dx,axis=1) # Calculate the mean of each trajectory dx.
    return traj_dx_mean


def acorr_dx(traj,norm=True):
    """ Calculates the velocity autocorrelation of displacements of given trajectories in numpy
    """
    traj_dx = traj[:,1:] - traj[:,0:-1] # Calculate trajectory dxs
    num_traj = traj_dx.shape[0] # Number of trajectories
    t_points = traj_dx.shape[1] # Number of time points in dx
    acorr = [] # Initialize autocorrelation array
    traj_dx_mean = np.mean(traj_dx,axis=1,keepdims=True) # Calculate the mean of each trajectory dx. Keep dimensions for broadcasting
    traj_dx_var = np.var(traj_dx,axis=1,keepdims=True) # Calculate the variance of each trajectory dx. Keep dimensions for broadcasting
    traj_dx_centered = (traj_dx - traj_dx_mean) # Center the trajectories by subtracting their means

    for i in range(1,t_points):
        traj_acorr_matrix = traj_dx_centered[:, i:]*traj_dx_centered[:, :-i] # Calculate a matrix of autocorrelation for each tau
        acorr.append(np.mean(traj_acorr_matrix, axis=1)) # Append the average of the autocorrelations at each tau to the acorr array
    acorr = np.stack(acorr,axis=1) # Make a 2D array from the list object
    if norm:
        acorr = acorr/traj_dx_var # Normalize the autocorrelations of each trajectory by the variance of the trajectory
    acorr = np.insert(acorr,0,np.ones([num_traj]),axis=1) # Insert a 1 at the beginning of the autocorrelation of each trajectory (correlation of a trajectory with itself is always 1)
    return acorr


def moments_dx_loss(train,decoded,moment):
    """ Calculates the mean squared displacement loss between the moments of the distribution of displacement of trajectories
        for LEONARDO's loss function
    """
    train_loss = moment(train)
    decoded_loss = moment(decoded)
    return ((train_loss-decoded_loss)**2).mean()


def kurtosis_dx_torch(traj_dx):
    """ Calculates the kurtosis of displacements of given trajectories in pytorch
    """
    traj_dx_mean = torch.mean(traj_dx, axis=1, keepdim=True) 
    traj_dx_centered = traj_dx - traj_dx_mean
    traj_dx_kurtosis_dist_num = traj_dx_centered**4
    traj_dx_kurtosis_dist_den = traj_dx_centered**2
    traj_dx_kurtosis = torch.mean(traj_dx_kurtosis_dist_num, axis=1) / torch.mean(traj_dx_kurtosis_dist_den, axis=1)**2
    return traj_dx_kurtosis


def skewness_dx_torch(traj_dx):
    """ Calculates the skewness of displacements of given trajectories in pytorch
    """
    traj_dx_mean = torch.mean(traj_dx, axis=1, keepdim=True) 
    traj_dx_centered = traj_dx - traj_dx_mean
    traj_dx_skewness_dist_num = traj_dx_centered**3
    traj_dx_skewness_dist_den = traj_dx_centered**2
    traj_dx_skewness = torch.mean(traj_dx_skewness_dist_num, axis=1) / torch.mean(traj_dx_skewness_dist_den, axis=1)**1.5
    return traj_dx_skewness


def variance_dx_torch(traj_dx):
    """ Calculates the variance of displacements of given trajectories in pytorch
    """
    traj_dx_mean = torch.mean(traj_dx, axis=1, keepdim=True)
    traj_dx_centered = traj_dx - traj_dx_mean
    traj_dx_variance = torch.mean(traj_dx_centered**2, axis=1)
    return traj_dx_variance

def mean_dx_torch(traj_dx):
    """ Calculates the mean of displacements of given trajectories in pytorch
    """
    traj_dx_mean = torch.mean(traj_dx,axis=1)
    return traj_dx_mean


def autocorrelation_dx_torch(traj_dx):
    """ Calculates the velocity autocorrelation of displacements of given trajectories in pytorch
    """
    acorr_dx = autocorrelation(traj_dx,1)
    return acorr_dx


def autocorrelation_dx_distribution(traj_dx,tau):
    """ Calculates the autocorrelation distribution of given trajectories in pytorch
    """
    n = traj_dx.shape[0]
    m = traj_dx.shape[1]
    traj_dx_mean = torch.mean(traj_dx,axis=1,keepdims=True)
    traj_dx_var = torch.var(traj_dx,axis=1,keepdims=True)
    traj_dx_norm = (traj_dx - traj_dx_mean)
    traj_acorr_matrix_3D = torch.empty([tau,n,m-1],device=device)
    for i in range(1,tau+1):
        traj_acorr_matrix = traj_dx_norm[:, i:]*traj_dx_norm[:, :-i]
        traj_acorr_matrix_3D[i-1] = torch.cat([torch.zeros([n,i-1],device=device),traj_acorr_matrix],axis=1) 
    traj_acorr_matrix_3D = traj_acorr_matrix_3D/traj_dx_var
    return traj_acorr_matrix_3D


def variance_acorr(acorrs):
    """ Calculates the variance of autocorrelations of displacements of given trajectories in pytorch
    """
    var = torch.var(acorrs,axis=1).unsqueeze(0)
    return var


def mean_jump_loss(img_dx,decoded_dx):
    """ Calculates the mean jump loss for LEONARDO's loss function
    """
    mean_jump = torch.mean(torch.absolute(decoded_dx),axis=0)
    mean_jump_trainset = torch.mean(torch.absolute(img_dx),axis=0)
    mean_jump_trainset_mean = torch.mean(mean_jump_trainset,axis=0,keepdim=True)
    mean_jump_diff_squared = (mean_jump - mean_jump_trainset_mean)**2
    mean_jump_diff_squared_mean = torch.mean(mean_jump_diff_squared)
    return mean_jump_diff_squared_mean

