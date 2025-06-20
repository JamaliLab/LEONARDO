"""
utils.py
This script contains functions for generating diffusion simulations, 
data generators needed for the network training/testing, and other necessary 
functions.
Original version by Granik et al is accessible at:  https://github.com/AnomDiffDB/DB
Updated version of this function has bugs fixed on the standard deviation of the data 
generated from different classes of diffusions, heavy-tailed distribution of waiting times
 in a CTRW class, and new functions are added to simulate hybrid trajectories 
"""


import numpy as np
from scipy import stats
import tensorflow as tf
import random
from models_theory import models_theory
from tensorflow.keras.utils import to_categorical

def set_random_seed(seed):
    """
    Function set_random_seed sets the random seed for different packages used in the utils file.

    Input:
        seed - numeric value of random seed
    
    Output:
        None. The random seed is set within the function
    """

    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


set_random_seed(21) # Set random seed



def Sub_brownian(x0, n, dt, delta, out=None):
    x0 = np.asarray(x0)
    # generate a sample of n numbers from a normal distribution.
    r = stats.norm.rvs(size=x0.shape + (n,), scale=delta*np.sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)
    # Compute Brownian motion by forming the cumulative sum of random samples. 
    np.cumsum(r, axis=-1, out=out)
    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out

def Brownian(N=1000,T=50,delta=1):
    """
    Brownian - generate Brownian motion trajectory (x,y)

    Inputs: 
        N - number of points to generate
        T - End time 
        delta - Diffusion coefficient

    Outputs:
        out1 - x axis values for each point of the trajectory
        out2 - y axis values for each point of the trajectory
    """

    x = np.empty((2,N+1))
    x[:, 0] = 0.0
    
    Sub_brownian(x[:,0], N, T/N, delta, out=x[:,1:])
    
    out1 = x[0]
    out2 = x[1]
    
    return out1,out2





def normalize_trajectory(traj_x, traj_y):
    # Center the trajectories by subtracting their respective means
    traj_x -= np.min(traj_x)
    traj_y -= np.min(traj_y)
    
    # Compute the global range (max value of either x or y - min value of either x or y)
    global_max = max(np.max(traj_x), np.max(traj_y))
    global_min = min(np.min(traj_x), np.min(traj_y))
    global_range = global_max - global_min
    
    # Avoid division by zero
    if global_range != 0:
        traj_x /= global_range
        traj_y /= global_range
    
    return np.diff(traj_x), np.diff(traj_y)


    # Modified generate function

def generate_classification(batchsize, steps, dim, alpha_fbm_lims=[0.1, 1], alpha_ctrw_lims=[0.1, 1], alpha_lw_lims=[1, 2], alpha_sbm_lims=[0.1, 1], alpha_attm_lims=[0.1, 1], include_lptem=None):
    num_classes = 6  # Only the original six classes
    samples_per_class = batchsize // num_classes

    # Initialize the dataset
    x = np.zeros((batchsize, steps - 1, dim))
    y = np.zeros((batchsize,))

    # Track the number of samples per class
    class_counts = [0] * num_classes  
    i = 0  # Initialize batch index

    while i < batchsize:
        for trajectory_type in range(num_classes):
            if class_counts[trajectory_type] < samples_per_class:
                
                if trajectory_type == 0:  # Brownian motion
                    bm_trajectory_x, bm_trajectory_y = Brownian(N=steps - 1)
                    trajectory_x, trajectory_y = normalize_trajectory(bm_trajectory_x, bm_trajectory_y)
                
                elif trajectory_type == 1:  # FBM
                    alpha_FBM = np.random.uniform(alpha_fbm_lims[0], alpha_fbm_lims[1])
                    traj_fbm = models_theory().fbm(alpha=alpha_FBM, T=steps, D=2)
                    trajectory_x, trajectory_y = normalize_trajectory(traj_fbm[:steps], traj_fbm[steps:])
                
                elif trajectory_type == 2:  # CTRW
                    alpha_CTRW = np.random.uniform(alpha_ctrw_lims[0], alpha_ctrw_lims[1])
                    traj_ctrw = models_theory().ctrw(alpha=alpha_CTRW, T=steps, D=2)
                    trajectory_x, trajectory_y = normalize_trajectory(traj_ctrw[:steps], traj_ctrw[steps:])
                
                elif trajectory_type == 3:  # LW
                    alpha_lw = np.random.uniform(alpha_lw_lims[0], alpha_lw_lims[1])
                    traj_lw = models_theory().lw(alpha=alpha_lw, T=steps, D=2)
                    trajectory_x, trajectory_y = normalize_trajectory(traj_lw[:steps], traj_lw[steps:])
                
                elif trajectory_type == 4:  # SBM
                    alpha_sbm = np.random.uniform(alpha_sbm_lims[0], alpha_sbm_lims[1])
                    traj_sbm = models_theory().sbm(alpha=alpha_sbm, T=steps, D=2)
                    trajectory_x, trajectory_y = normalize_trajectory(traj_sbm[:steps], traj_sbm[steps:])
                
                elif trajectory_type == 5:  # ATTM
                    alpha_attm = np.random.uniform(alpha_attm_lims[0], alpha_attm_lims[1])
                    traj_attm = models_theory().attm(alpha=alpha_attm, T=steps, D=2)
                    trajectory_x, trajectory_y = normalize_trajectory(traj_attm[:steps], traj_attm[steps:])

                # Store the generated trajectory and label
                x[i, :, 0] = trajectory_x
                x[i, :, 1] = trajectory_y
                y[i] = trajectory_type
                class_counts[trajectory_type] += 1
                i += 1  # Move to the next batch index

    # If include_lptem is provided, normalize and append it to the generated dataset
    if include_lptem is not None:
        num_lptem = include_lptem.shape[0]
        normalized_lptem = np.zeros((num_lptem, steps - 1, dim))
        
        for j in range(num_lptem):
            traj_x, traj_y = include_lptem[j, :, 0], include_lptem[j, :, 1]
            norm_x, norm_y = normalize_trajectory(traj_x, traj_y)
            normalized_lptem[j, :, 0] = norm_x
            normalized_lptem[j, :, 1] = norm_y

        lptem_labels = np.full(num_lptem, 6)  # Assign label 6 for LPTEM trajectories
        x = np.concatenate((x, normalized_lptem), axis=0)
        y = np.concatenate((y, lptem_labels), axis=0)

    # Shuffle the combined dataset
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # Convert the labels to one-hot encoding for categorical cross-entropy loss
    y_one_hot = to_categorical(y, num_classes=(num_classes + 1) if include_lptem is not None else num_classes)

    return x, y_one_hot





def generate_brownian(batchsize, steps, dim):
    x = np.zeros((batchsize, steps - 1, dim))
    for i in range(batchsize):
        bm_trajectory_x, bm_trajectory_y = Brownian(N=steps - 1)
        trajectory_x, trajectory_y = normalize_trajectory(bm_trajectory_x, bm_trajectory_y)       
        x[i, :, 0] = trajectory_x
        x[i, :, 1] = trajectory_y
    return x

def generate_fbm(batchsize, steps, dim, alpha_fbm_lims=[0.1, 1]):
    x = np.zeros((batchsize, steps - 1, dim))
    for i in range(batchsize):
        alpha_FBM = np.random.uniform(alpha_fbm_lims[0], alpha_fbm_lims[1])
        traj_fbm = models_theory().fbm(alpha=alpha_FBM, T=steps, D=2)
        trajectory_x, trajectory_y = normalize_trajectory(traj_fbm[:steps], traj_fbm[steps:])
        x[i, :, 0] = trajectory_x
        x[i, :, 1] = trajectory_y
    return x

def generate_ctrw(batchsize, steps, dim, alpha_ctrw_lims=[0.1, 1]):
    x = np.zeros((batchsize, steps - 1, dim))
    for i in range(batchsize):
        alpha_CTRW = np.random.uniform(alpha_ctrw_lims[0], alpha_ctrw_lims[1])
        traj_ctrw = models_theory().ctrw(alpha=alpha_CTRW, T=steps, D=2)
        trajectory_x, trajectory_y = normalize_trajectory(traj_ctrw[:steps], traj_ctrw[steps:])
        x[i, :, 0] = trajectory_x
        x[i, :, 1] = trajectory_y
    return x

def generate_lw(batchsize, steps, dim, alpha_lw_lims=[1, 2]):
    x = np.zeros((batchsize, steps - 1, dim))
    for i in range(batchsize):
        alpha_lw = np.random.uniform(alpha_lw_lims[0], alpha_lw_lims[1])
        traj_lw = models_theory().lw(alpha=alpha_lw, T=steps, D=2)
        trajectory_x, trajectory_y = normalize_trajectory(traj_lw[:steps], traj_lw[steps:])
        x[i, :, 0] = trajectory_x
        x[i, :, 1] = trajectory_y
    return x

def generate_sbm(batchsize, steps, dim, alpha_sbm_lims=[0.1, 1]):
    x = np.zeros((batchsize, steps - 1, dim))
    for i in range(batchsize):
        alpha_sbm = np.random.uniform(alpha_sbm_lims[0], alpha_sbm_lims[1])
        traj_sbm = models_theory().sbm(alpha=alpha_sbm, T=steps, D=2)
        trajectory_x, trajectory_y = normalize_trajectory(traj_sbm[:steps], traj_sbm[steps:])
        x[i, :, 0] = trajectory_x
        x[i, :, 1] = trajectory_y
    return x

def generate_attm(batchsize, steps, dim, alpha_attm_lims=[0.1, 1]):
    x = np.zeros((batchsize, steps - 1, dim))
    for i in range(batchsize):
        alpha_attm = np.random.uniform(alpha_attm_lims[0], alpha_attm_lims[1])
        traj_attm = models_theory().attm(alpha=alpha_attm, T=steps, D=2)
        trajectory_x, trajectory_y = normalize_trajectory(traj_attm[:steps], traj_attm[steps:])
        x[i, :, 0] = trajectory_x
        x[i, :, 1] = trajectory_y
    return x
