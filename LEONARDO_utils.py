import numpy as np    
import torch
import sys
import pathlib
import pandas as pd
# sys.path.append('../Trajectory_simulation/lib/')
from pytorch_forecasting.utils import autocorrelation
import os
import random
from IPython.display import display, HTML
from scipy.stats import kurtosis

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
    """Normalizes 2D trajectories between 0 and 1.
    
    Args:
        trajs (ndarray): Input trajectories of shape (batch_size, length, 2).
        
    Returns:
        new_trajs (ndarray): Normalized trajectories.
    """
    loc_min = trajs.min(axis=1, keepdims=True)
    xy_centered = trajs - loc_min
    glob_min = xy_centered.min(axis=1, keepdims=True).min(axis=2, keepdims=True)
    glob_max = xy_centered.max(axis=1, keepdims=True).max(axis=2, keepdims=True)
    xy_normalized = (xy_centered) / (glob_max - glob_min)

    return xy_normalized




def prepare_dataset(dataset, return_labels=False, n_labels=0, norm=True, shuffle=True, to_tensor=True, test=False):
    """ Normalizes, shuffles, and converts trajectories to torch tensors.
    
    Args:
        dataset (ndarray): Input dataset of trajectories.
        return_labels (bool): If True, return labels.
        n_labels (int): Number of label columns in the dataset.
        norm (bool): If True, normalize the dataset.
        shuffle (bool): If True, shuffle the dataset.
        to_tensor (bool): If True, convert to torch tensors.
        test (bool): If True, adjust for test dataset.
    
    Returns:
        dataset (torch.Tensor): Processed dataset.
        labels (ndarray): Labels if `return_labels` is True.
    """
    if shuffle:
        np.random.shuffle(dataset)
    if test:
        n_labels = 1

    labels = dataset[:, :n_labels]
    dataset = dataset[:, n_labels:]

    if norm:
        dataset = normalize(dataset)

    if to_tensor:
        dataset = torch.from_numpy(dataset).float()

    if return_labels:
        return dataset, labels

    return dataset



def calculate_r(traj):
    """ Calculates the radial component (r) of 2D trajectories.
    
    Args:
        traj (ndarray): Input trajectories of shape (batch_size, length, 2).
        
    Returns:
        r (ndarray): Radial component of the trajectory, shape (batch_size, length).
    """
    r = np.sqrt(traj[:, :, 0]**2 + traj[:, :, 1]**2)  # r = sqrt(x^2 + y^2)
    return r





def kurtosis_dx(r, norm=True):
    """ Calculates the kurtosis of radial displacements of given trajectories in numpy.
    
    Args:
        traj (ndarray): Input trajectories of shape (batch_size, length, 2).
        norm (bool): If True, normalize by variance.
        
    Returns:
        traj_r_kurtosis (ndarray): Kurtosis of radial displacements.
    """
    # r = calculate_r(traj)  # Compute radial component r
    r_dx = r[:, 1:] - r[:, :-1]  # Calculate radial displacements (dr)
    r_dx_mean = np.mean(r_dx, axis=1, keepdims=True)  # Mean of radial displacements
    r_dx_centered = r_dx - r_dx_mean  # Center the radial displacements
    
    r_dx_kurtosis_dist_num = r_dx_centered ** 4  # Fourth power of centered displacements
    r_dx_kurtosis_dist_den = r_dx_centered ** 2  # Second power of centered displacements
    
    if norm:
        traj_r_kurtosis = np.mean(r_dx_kurtosis_dist_num, axis=1) / np.mean(r_dx_kurtosis_dist_den, axis=1) ** 2
    else:
        traj_r_kurtosis = np.mean(r_dx_kurtosis_dist_num, axis=1)

    return traj_r_kurtosis





def skewness_dx(r, norm=True):
    """ Calculates the skewness of radial displacements of given trajectories in numpy.
    
    Args:
        traj (ndarray): Input trajectories of shape (batch_size, length, 2).
        norm (bool): If True, normalize by variance.
        
    Returns:
        traj_r_skewness (ndarray): Skewness of radial displacements.
    """
    # r = calculate_r(traj)  # Compute radial component r
    r_dx = r[:, 1:] - r[:, :-1]  # Calculate radial displacements (dr)
    r_dx_mean = np.mean(r_dx, axis=1, keepdims=True)  # Mean of radial displacements
    r_dx_centered = r_dx - r_dx_mean  # Center the radial displacements
    
    r_dx_skewness_dist_num = r_dx_centered ** 3  # Third power of centered displacements
    r_dx_skewness_dist_den = r_dx_centered ** 2  # Second power of centered displacements
    
    if norm:
        traj_r_skewness = np.mean(r_dx_skewness_dist_num, axis=1) / np.mean(r_dx_skewness_dist_den, axis=1) ** 1.5
    else:
        traj_r_skewness = np.mean(r_dx_skewness_dist_num, axis=1)

    return traj_r_skewness





def variance_dx(r):
    """ Calculates the variance of radial displacements of given trajectories in numpy.
    
    Args:
        traj (ndarray): Input trajectories of shape (batch_size, length, 2).
        
    Returns:
        traj_r_variance (ndarray): Variance of radial displacements.
    """
    # r = calculate_r(traj)  # Compute radial component r
    r_dx = r[:, 1:] - r[:, :-1]  # Calculate radial displacements (dr)
    r_dx_mean = np.mean(r_dx, axis=1, keepdims=True)  # Mean of radial displacements
    r_dx_centered = r_dx - r_dx_mean  # Center the radial displacements
    
    r_dx_variance = np.mean(r_dx_centered ** 2, axis=1)  # Calculate variance
    return r_dx_variance






def mean_dx(r):
    """ Calculates the mean of radial displacements of given trajectories in numpy.
    
    Args:
        traj (ndarray): Input trajectories of shape (batch_size, length, 2).
        
    Returns:
        traj_r_mean (ndarray): Mean radial displacement values.
    """
    # r = calculate_r(traj)  # Compute radial component r
    r_dx = r[:, 1:] - r[:, :-1]  # Calculate radial displacements (dr)
    r_dx_mean = np.mean(r_dx, axis=1)  # Mean of radial displacements
    return r_dx_mean






def acorr_dx(r, norm=True):
    """ Calculates the velocity autocorrelation of radial displacements of given trajectories in numpy.
    
    Args:
        traj (ndarray): Input trajectories of shape (batch_size, length, 2).
        norm (bool): If True, normalize autocorrelations.
        
    Returns:
        acorr (ndarray): Autocorrelation values for radial displacements.
    """
    # r = calculate_r(traj)  # Compute radial component r
    r_dx = r[:, 1:] - r[:, :-1]  # Calculate radial displacements (dr)
    
    num_traj = r_dx.shape[0]  # Number of trajectories
    t_points = r_dx.shape[1]  # Number of time points in dr

    acorr = []  # Initialize autocorrelation array
    r_dx_mean = np.mean(r_dx, axis=1, keepdims=True)  # Mean of radial displacements
    r_dx_var = np.var(r_dx, axis=1, keepdims=True)  # Variance of radial displacements
    r_dx_centered = r_dx - r_dx_mean  # Center the radial displacements

    for i in range(1, t_points):
        r_acorr_matrix = np.einsum('ij,ij->i', r_dx_centered[:, i:], r_dx_centered[:, :-i])  # Autocorrelation for each tau
        acorr.append(np.mean(r_acorr_matrix, axis=0))  # Append the mean autocorrelations

    acorr = np.stack(acorr, axis=1)  # Convert list to array
    if norm:
        acorr = acorr / r_dx_var  # Normalize by variance

    acorr = np.insert(acorr, 0, np.ones([num_traj]), axis=1)  # Insert 1 at the beginning (correlation with self)
    return acorr





def moments_loss(train, decoded, moment):
    """ Calculates the mean squared displacement loss between the moments of the distribution of displacement 
        for Davinci's loss function.
    
    Args:
        train (ndarray): Ground truth trajectories of shape (batch_size, length, 2).
        decoded (ndarray): Decoded trajectories of shape (batch_size, length, 2).
        moment (function): Function to calculate the moment (e.g., kurtosis, skewness).
        
    Returns:
        loss (float): The loss based on the difference between the moments of the train and decoded displacements.
    """
    train_loss = moment(train)
    decoded_loss = moment(decoded)
    return ((train_loss - decoded_loss) ** 2).mean()


def calculate_r_torch(traj):
    """ Calculates the radial component (r) of 2D trajectories in PyTorch.
    
    Args:
        traj (Tensor): Input trajectories of shape (batch_size, length, 2).
        
    Returns:
        r (Tensor): Radial component of the trajectory, shape (batch_size, length).
    """
    r = torch.sqrt(traj[:, :, 0]**2 + traj[:, :, 1]**2)  # r = sqrt(x^2 + y^2)
    return r







def kurtosis_torch(signal):
    """ Calculates the kurtosis of displacements of given signal in PyTorch.
    
    Args:
        signal (Tensor): Input signal of shape (batch_size, length, 2).
        
    Returns:
        signal_kurtosis (Tensor): Kurtosis of displacements.
    """
   
    # signal = signal[:, 1:] - signal[:, :-1]  # Calculate displacements
    signal_mean = torch.mean(signal, axis=1, keepdim=True)  # Mean of displacements

    signal_centered = signal - signal_mean  # Center the displacements
    
    signal_kurtosis_dist_num = signal_centered ** 4  # Fourth power of centered displacements
    signal_kurtosis_dist_den = signal_centered ** 2  # Second power of centered displacements
    
    signal_kurtosis = torch.mean(signal_kurtosis_dist_num, axis=1) / torch.mean(signal_kurtosis_dist_den, axis=1) ** 2

    return signal_kurtosis







def skewness_torch(signal):
    """ Calculates the skewness of displacements of given signal in PyTorch.
    
    Args:
        signal (Tensor): Input signal of shape (batch_size, length, 2).
        
    Returns:
        signal_skewness (Tensor): Skewness of displacements.
    """
    # signal = signal[:, 1:] - signal[:, :-1]  # Calculate displacements
    signal_mean = torch.mean(signal, axis=1, keepdim=True)  # Mean of displacements
    signal_centered = signal - signal_mean  # Center the displacements
    
    signal_skewness_dist_num = signal_centered ** 3  # Third power of centered displacements
    signal_skewness_dist_den = signal_centered ** 2  # Second power of centered displacements
    
    signal_skewness = torch.mean(signal_skewness_dist_num, axis=1) / torch.mean(signal_skewness_dist_den, axis=1) ** 1.5

    return signal_skewness






def variance_torch(signal):
    """ Calculates the variance of displacements of given signal in PyTorch.
    
    Args:
        signal (Tensor): Input signal of shape (batch_size, length, 2).
        
    Returns:
        signal_variance (Tensor): Variance of displacements.
    """
   
    # signal = signal[:, 1:] - signal[:, :-1]  # Calculate displacements
    signal_mean = torch.mean(signal, axis=1, keepdim=True)  # Mean of displacements
    signal_centered = signal - signal_mean  # Center the displacements
    
    signal_variance = torch.mean(signal_centered ** 2, axis=1)  # Calculate variance
    return signal_variance







def mean_torch(signal):
    """ Calculates the mean of displacements of given signalectories in PyTorch.
    
    Args:
        signal (Tensor): Input signalectories of shape (batch_size, length, 2).
        
    Returns:
        signal_mean (Tensor): Mean displacement values.
    """
    # signal_dx = signal[:, 1:] - signal[:, :-1]  # Calculate displacements

    signal_mean = torch.mean(signal, axis=1)  # Mean of displacements
    # print(signal_mean.shape)
    return signal_mean



def median_torch(signal):
    """ Calculates the median of displacements of given signalectories in PyTorch.
    
    Args:
        signal (Tensor): Input signalectories of shape (batch_size, length, 2).
        
    Returns:
        signal_median (Tensor): median displacement values.
    """
    # signal_dx = signal[:, 1:] - signal[:, :-1]  # Calculate displacements

    signal_median = torch.quantile(signal, 0.5, dim=1)  # median of displacements
    # print(signal_median.shape)
    return signal_median







def autocorrelation_dx_torch(traj):
    """ Calculates the velocity autocorrelation of radial displacements of given trajectories in PyTorch.
    
    Args:
        traj (Tensor): Input trajectories of shape (batch_size, length, 2).
        
    Returns:
        acorr (Tensor): Autocorrelation values for radial displacements.
    """
    
    traj_dx = traj[:, 1:] - traj[:, :-1]  # Calculate displacements
    
    acorr_dx = autocorrelation(traj_dx, 1)  # Use autocorrelation function for the radial displacements
    # print('Acorr shape: ',acorr_dx.shape)
    return acorr_dx



def autocorrelation_pos_torch(traj):
    """ Calculates the velocity autocorrelation of radial displacements of given trajectories in PyTorch.
    
    Args:
        traj (Tensor): Input trajectories of shape (batch_size, length, 2).
        
    Returns:
        acorr (Tensor): Autocorrelation values for radial displacements.
    """
    
    acorr_pos = autocorrelation(traj, 1)  # Use autocorrelation function for the radial displacements
    # print('Acorr shape: ',acorr_pos.shape)
    return acorr_pos





def calculate_correlation_torch(jumps):
    """
    Calculates the correlation coefficient between x and y jumps
    for each trajectory in a batch using the formula from torch.corrcoef.
    
    Args:
        trajectories (torch.Tensor): A 3D tensor of shape (num_traj, seq_len, 2),
                                     where the last dimension represents (x, y) coordinates.
    
    Returns:
        torch.Tensor: A 1D tensor of shape (num_traj,) containing the correlation
                      coefficients for each trajectory.
    """

    # Extract x and y jumps
    x_jumps = jumps[..., 0]  # Shape: (num_traj, seq_len)
    y_jumps = jumps[..., 1]  # Shape: (num_traj, seq_len)

    # Compute means
    mean_x = x_jumps.mean(dim=1, keepdim=True)
    mean_y = y_jumps.mean(dim=1, keepdim=True)

    # Center the data (subtract means)
    x_centered = x_jumps - mean_x
    y_centered = y_jumps - mean_y

    # Compute covariance: Cov(X, Y) = mean((X - mean(X)) * (Y - mean(Y)))
    covariance = (x_centered * y_centered).mean(dim=1)

    # Compute variances: Var(X) = mean((X - mean(X))^2), Var(Y) = mean((Y - mean(Y))^2)
    variance_x = (x_centered ** 2).mean(dim=1)
    variance_y = (y_centered ** 2).mean(dim=1)

    # Compute the correlation coefficient: Cov(X, Y) / sqrt(Var(X) * Var(Y))
    correlation = covariance / torch.sqrt(variance_x * variance_y)

    return correlation





def display_model_properties(model):
    """ Displays the loaded model's properties in a neat tabular format with subheadings for loss weights. """
    model_dict = model.model_dict

    # Extract the last epoch values for each loss component
    properties = {
        'Trajectory Length': model_dict.get('Trajectory Length', 'N/A'),
        'Latent Space': model_dict.get('Latent Space', 'N/A'),
        'Epochs': model_dict.get('Epochs', 'N/A'),
        'Transformer Embed Size': model_dict['Model Architecture']['Transformer Architecture']['embed size'],
        'Transformer  Number of Layers': model_dict['Model Architecture']['Transformer Architecture']['num layers'],
        'Transformer  Heads': model_dict['Model Architecture']['Transformer Architecture']['heads'],
        'Transformer  Forward Expansion': model_dict['Model Architecture']['Transformer Architecture']['forward expansion'],
        'Bottleneck Encoding Conv1': model_dict['Model Architecture']['Bottleneck Architecture']['encoding']['conv1'],
        'Bottleneck Encoding Conv2': model_dict['Model Architecture']['Bottleneck Architecture']['encoding']['conv2'],
        'Bottleneck Encoding Linear1': model_dict['Model Architecture']['Bottleneck Architecture']['encoding']['linear1'],
        'Bottleneck Decoding Linear1': model_dict['Model Architecture']['Bottleneck Architecture']['decoding']['linear1'],
        'Bottleneck Decoding ConvTranspose1': model_dict['Model Architecture']['Bottleneck Architecture']['decoding']['convtranspose1'],
        'Final Training Loss': model.training_loss_per_epoch[-1] if len(model.training_loss_per_epoch) > 0 else 'N/A',
        'Final Validation Loss': model.validation_loss_per_epoch[-1] if len(model.validation_loss_per_epoch) > 0 else 'N/A',
        'Final Training MSE Loss': model.training_mse_loss_per_epoch[-1] if len(model.training_mse_loss_per_epoch) > 0 else 'N/A',
        'Final Validation MSE Loss': model.validation_mse_loss_per_epoch[-1] if len(model.validation_mse_loss_per_epoch) > 0 else 'N/A',
        'Final Training KL Loss': model.training_kl_loss_per_epoch[-1] if len(model.training_kl_loss_per_epoch) > 0 else 'N/A',
        'Final Validation KL Loss': model.validation_kl_loss_per_epoch[-1] if len(model.validation_kl_loss_per_epoch) > 0 else 'N/A',
        'Final Training Kurtosis Loss': model.training_kurtosis_loss_per_epoch[-1] if len(model.training_kurtosis_loss_per_epoch) > 0 else 'N/A',
        'Final Validation Kurtosis Loss': model.validation_kurtosis_loss_per_epoch[-1] if len(model.validation_kurtosis_loss_per_epoch) > 0 else 'N/A',
        'Final Training Skewness Loss': model.training_skewness_loss_per_epoch[-1] if len(model.training_skewness_loss_per_epoch) > 0 else 'N/A',
        'Final Validation Skewness Loss': model.validation_skewness_loss_per_epoch[-1] if len(model.validation_skewness_loss_per_epoch) > 0 else 'N/A',
        'Final Training Variance Loss': model.training_variance_loss_per_epoch[-1] if len(model.training_variance_loss_per_epoch) > 0 else 'N/A',
        'Final Validation Variance Loss': model.validation_variance_loss_per_epoch[-1] if len(model.validation_variance_loss_per_epoch) > 0 else 'N/A',
        'Final Training Mean Loss': model.training_mean_loss_per_epoch[-1] if len(model.training_mean_loss_per_epoch) > 0 else 'N/A',
        'Final Validation Mean Loss': model.validation_mean_loss_per_epoch[-1] if len(model.validation_mean_loss_per_epoch) > 0 else 'N/A',
        'Final Training Median Loss': model.training_median_loss_per_epoch[-1] if len(model.training_median_loss_per_epoch) > 0 else 'N/A',
        'Final Validation Medain Loss': model.validation_median_loss_per_epoch[-1] if len(model.validation_median_loss_per_epoch) > 0 else 'N/A',
        'Final Training Autocorrelation Loss': model.training_acorr_loss_per_epoch[-1] if len(model.training_acorr_loss_per_epoch) > 0 else 'N/A',
        'Final Validation Autocorrelation Loss': model.validation_acorr_loss_per_epoch[-1] if len(model.validation_acorr_loss_per_epoch) > 0 else 'N/A',
        'Final Training Autocorrelation Batch Loss': model.training_acorr_batch_loss_per_epoch[-1] if len(model.training_acorr_batch_loss_per_epoch) > 0 else 'N/A',
        'Final Validation Autocorrelation Batch Loss': model.validation_acorr_batch_loss_per_epoch[-1] if len(model.validation_acorr_batch_loss_per_epoch) > 0 else 'N/A',
        'Final Training xy-Correlation Loss': model.training_xycorr_loss_per_epoch[-1] if len(model.training_xycorr_loss_per_epoch) > 0 else 'N/A',
        'Final Validation xy-Correlation Loss': model.validation_xycorr_loss_per_epoch[-1] if len(model.validation_xycorr_loss_per_epoch) > 0 else 'N/A',
        'Final Training Positional Acorr Loss': model.training_pos_acorr_loss_per_epoch[-1] if len(model.training_pos_acorr_loss_per_epoch) > 0 else 'N/A',
        'Final Validation Positional Acorr Loss': model.validation_pos_acorr_loss_per_epoch[-1] if len(model.validation_pos_acorr_loss_per_epoch) > 0 else 'N/A',
        
    }

    # Extract loss weights with subheadings
    loss_weights = model_dict.get('loss weights', {})
    for key, value in loss_weights.items():
        properties[f'Loss Weight - {key}'] = value

    # Include additional training parameters
    training_parameters = {
        'Batch Size': model_dict.get('Batch Size', 'N/A'),
        'Learning Rate': model_dict.get('Learning Rate', 'N/A'),
        'Weight Decay': model_dict.get('Weight Decay', 'N/A'),
        'Optimizer': model_dict.get('Optimizer', 'N/A'),
        'Trajectory Types': model_dict.get('Trajectory Types', 'N/A'),
        'Training Dataset Size': model_dict.get('Training Dataset Size', 'N/A'),
        'Validation Dataset Size': model_dict.get('Validation Dataset Size', 'N/A'),
        'Random Seed': model_dict.get('Random Seed', 'N/A'),
        'GPU number': model_dict.get('GPU number', 'N/A'),
        'Trained on dx or x?': model_dict.get('Trained on dx or x?', 'N/A'),
        'Normalization type': model_dict.get('Normalization type', 'N/A'),
        'Notes': model_dict.get('Notes', 'No notes')
    }

    # Merge all properties into a single dictionary for display
    all_properties = {**properties, **training_parameters}

    # Convert the properties to a pandas DataFrame for a clean display
    df = pd.DataFrame(all_properties.items(), columns=['Property', 'Value'])
    
    # Adjust the display to wrap text using HTML
    display(HTML(df.to_html(index=False).replace('\\n', '<br>')))



