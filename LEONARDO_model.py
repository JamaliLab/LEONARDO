import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
import os
from LEONARDO_utils import kurtosis_dx_torch, skewness_dx_torch, variance_dx_torch, mean_dx_torch, autocorrelation_dx_torch, \
                  autocorrelation_dx_distribution, variance_acorr, mean_jump_loss, moments_dx_loss


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


device = 'cuda:3'


class LEONARDO(nn.Module):
    def __init__(self, length_trajectory, latent_space):
        super(LEONARDO, self).__init__()

        """ A LEONARDO object is created with the length of the trajectories and the size of the latent space as attributes.
            This object is used in the TransformerVAE model architecture.
        """

        ## Initialize attributes
        self.length_trajectory = length_trajectory
        self.latent_space = latent_space
        
        ## Initialize model training and validation losses
        self.training_loss_per_epoch = np.array([])
        self.training_kl_loss_per_epoch = np.array([])
        self.training_mse_loss_per_epoch = np.array([])
        self.training_kurtosis_loss_per_epoch = np.array([])
        self.training_skewness_loss_per_epoch = np.array([])
        self.training_variance_loss_per_epoch = np.array([])
        self.training_mean_loss_per_epoch = np.array([])
        self.training_acorr_loss_per_epoch = np.array([])
        self.training_mean_jump_loss_per_epoch = np.array([])
        self.training_acorr_batch_loss_per_epoch = np.array([])
        self.training_acorr_percentiles_loss_per_epoch = np.array([])
        self.validation_loss_per_epoch = np.array([])
        self.validation_kl_loss_per_epoch = np.array([])
        self.validation_mse_loss_per_epoch = np.array([])
        self.validation_kurtosis_loss_per_epoch = np.array([])
        self.validation_skewness_loss_per_epoch = np.array([])
        self.validation_variance_loss_per_epoch = np.array([])
        self.validation_mean_loss_per_epoch = np.array([])
        self.validation_acorr_loss_per_epoch = np.array([])
        self.validation_mean_jump_loss_per_epoch = np.array([])
        self.validation_acorr_batch_loss_per_epoch = np.array([])
        self.validation_acorr_percentiles_loss_per_epoch = np.array([])

        ## Initialize epochs
        self.epochs = 0
    
    
    def train_LEONARDO(self, epochs, dataloader, optimizer, device, val_loader = None):
        """ Trains LEONARDO, storing the training and validation losses at each epoch
        """
        total_epochs = self.epochs + epochs

        for epoch in range(epochs):
            # Train data
            loss_epoch, loss_kl_training, loss_mse_training, loss_kurtosis_training, loss_skewness_training, \
                loss_variance_training, loss_mean_training, loss_acorr_training, \
                    loss_mean_jump_training, loss_acorr_batch_training, loss_acorr_percentiles_training, dataset_size = (0,0,0,0,0,0,0,0,0,0,0,0)
            for data in dataloader:
                train_data = Variable(data).to(device)
                train_data_dx = train_data.squeeze()[:,1:]-train_data.squeeze()[:,0:-1]

                _, z_mean, z_log_var, decoded = self(train_data)
                decoded_dx = decoded.squeeze()[:,1:]-decoded.squeeze()[:,0:-1]
                kl_div = -0.5*torch.sum(1 + z_log_var 
                                          - z_mean**2 
                                          - torch.exp(z_log_var),
                                          axis = 1
                                          )
                kl_div = kl_div.mean()

                ## Calculating loss function components:

                # Four moments of the distribution of displacements
                kurtosis_difference_squared_mean = moments_dx_loss(train_data_dx,decoded_dx,kurtosis_dx_torch)
                skewness_difference_squared_mean = moments_dx_loss(train_data_dx,decoded_dx,skewness_dx_torch)
                variance_difference_squared_mean = moments_dx_loss(train_data_dx,decoded_dx,variance_dx_torch)
                mean_difference_squared_mean = moments_dx_loss(train_data_dx,decoded_dx,mean_dx_torch)
                
                # Velocity autocorrelation mean
                train_data_dx_acorr = autocorrelation_dx_torch(train_data_dx)[:,0:50]
                decoded_dx_acorr = autocorrelation_dx_torch(decoded_dx)[:,0:50]
                acorr_difference_squared_mean = ((decoded_dx_acorr-train_data_dx_acorr)**2).mean()

                # Velocity autocorrelation variance
                train_data_dx_acorr_distributions = autocorrelation_dx_distribution(train_data_dx,50)
                train_data_dx_acorr_distributions_variance = variance_acorr(train_data_dx_acorr_distributions)
                decoded_dx_acorr_distributions = autocorrelation_dx_distribution(decoded_dx,50)
                decoded_dx_acorr_distributions_variance = variance_acorr(decoded_dx_acorr_distributions)
                acorr_distributions_variance_difference_squared_mean = ((train_data_dx_acorr_distributions_variance - decoded_dx_acorr_distributions_variance)**2).mean()

                # Batch velocity autocorrelation
                train_data_dx_acorr_batch = autocorrelation_dx_torch(train_data_dx)
                decoded_dx_acorr_batch = autocorrelation_dx_torch(decoded_dx)
                acorr_batch_mean_difference_squared = ((decoded_dx_acorr_batch.mean(axis=0)-train_data_dx_acorr_batch.mean(axis=0))**2).mean()

                # Mean jump
                mean_jump_difference_squared_mean = mean_jump_loss(train_data_dx,decoded_dx)

                # Reconstruction loss
                mse = ((decoded_dx - train_data_dx)**2).mean()

                # Weighting factors of each loss term
                kurt_factor = 0.006
                skew_factor = 1
                var_factor = 500
                mean_factor = 100000
                acorr_factor = 100
                mse_factor = 3
                kl_factor = 0.01
                acorr_batch_factor = 1500
                mean_jump_factor = 100
                acorr_distributions_variance_factor = 10

                # Loss function
                loss = mse_factor*mse + kl_factor*kl_div + kurt_factor*kurtosis_difference_squared_mean\
                      + skew_factor*skewness_difference_squared_mean + var_factor*variance_difference_squared_mean\
                      + mean_factor*mean_difference_squared_mean + acorr_factor*acorr_difference_squared_mean\
                      + acorr_batch_factor*acorr_batch_mean_difference_squared + mean_jump_factor*mean_jump_difference_squared_mean\
                      + acorr_distributions_variance_factor*acorr_distributions_variance_difference_squared_mean
                      


                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update losses
                loss_epoch += loss.item() * len(data)
                loss_kl_training += kl_div.item() * len(data)
                loss_mse_training += mse.item() * len(data)
                loss_kurtosis_training += kurtosis_difference_squared_mean.item() * len(data)
                loss_skewness_training += skewness_difference_squared_mean.item() * len(data)
                loss_variance_training += variance_difference_squared_mean.item() * len(data)
                loss_mean_training += mean_difference_squared_mean.item() * len(data)
                loss_acorr_training += acorr_difference_squared_mean.item() * len(data)
                loss_mean_jump_training += mean_jump_difference_squared_mean.item() * len(data)
                loss_acorr_batch_training += acorr_batch_mean_difference_squared.item() * len(data)
                loss_acorr_percentiles_training += acorr_distributions_variance_difference_squared_mean.item() * len(data)
                dataset_size += len(data)
            
            #Append losses to their arrays
            self.training_loss_per_epoch = np.append(self.training_loss_per_epoch, loss_epoch/dataset_size)
            self.training_kl_loss_per_epoch = np.append(self.training_kl_loss_per_epoch, loss_kl_training/dataset_size)
            self.training_mse_loss_per_epoch = np.append(self.training_mse_loss_per_epoch, loss_mse_training/dataset_size)
            self.training_kurtosis_loss_per_epoch = np.append(self.training_kurtosis_loss_per_epoch, loss_kurtosis_training/dataset_size)
            self.training_skewness_loss_per_epoch = np.append(self.training_skewness_loss_per_epoch, loss_skewness_training/dataset_size)
            self.training_variance_loss_per_epoch = np.append(self.training_variance_loss_per_epoch, loss_variance_training/dataset_size)
            self.training_mean_loss_per_epoch = np.append(self.training_mean_loss_per_epoch, loss_mean_training/dataset_size)
            self.training_acorr_loss_per_epoch = np.append(self.training_acorr_loss_per_epoch, loss_acorr_training/dataset_size)
            self.training_mean_jump_loss_per_epoch = np.append(self.training_mean_jump_loss_per_epoch, loss_mean_jump_training/dataset_size)
            self.training_acorr_batch_loss_per_epoch = np.append(self.training_acorr_batch_loss_per_epoch, loss_acorr_batch_training/dataset_size)
            self.training_acorr_percentiles_loss_per_epoch = np.append(self.training_acorr_percentiles_loss_per_epoch, loss_acorr_percentiles_training/dataset_size)

            ## Validation
            if val_loader:
                val_loss_epoch, valset_size, loss_kl_val, loss_mse_val, loss_kurtosis_val, loss_skewness_val, \
                    loss_variance_val, loss_mean_val, \
                        loss_acorr_val, loss_mean_jump_val, loss_acorr_batch_val, loss_acorr_percentiles_val = (0,0,0,0,0,0,0,0,0,0,0,0)
                for data in val_loader:
                    train_data_dx = train_data.squeeze()[:,1:]-train_data.squeeze()[:,0:-1]
            

                    _, z_mean, z_log_var, decoded = self(train_data)
                    decoded_dx = decoded.squeeze()[:,1:]-decoded.squeeze()[:,0:-1]

                    kl_div = -0.5 * torch.sum(1 + z_log_var 
                                          - z_mean**2 
                                          - torch.exp(z_log_var),
                                          axis = 1
                                            )
                    kl_div = kl_div.mean()


                    ## Calculating loss function components:

                    # Four moments of the distribution of displacements
                    kurtosis_difference_squared_mean = moments_dx_loss(train_data_dx,decoded_dx,kurtosis_dx_torch)
                    skewness_difference_squared_mean = moments_dx_loss(train_data_dx,decoded_dx,skewness_dx_torch)
                    variance_difference_squared_mean = moments_dx_loss(train_data_dx,decoded_dx,variance_dx_torch)
                    mean_difference_squared_mean = moments_dx_loss(train_data_dx,decoded_dx,mean_dx_torch)
                    
                    # Velocity autocorrelation mean
                    train_data_dx_acorr = autocorrelation_dx_torch(train_data_dx)[:,0:50]
                    decoded_dx_acorr = autocorrelation_dx_torch(decoded_dx)[:,0:50]
                    acorr_difference_squared_mean = ((decoded_dx_acorr-train_data_dx_acorr)**2).mean()

                    # Velocity autocorrelation variance
                    train_data_dx_acorr_distributions = autocorrelation_dx_distribution(train_data_dx,50)
                    train_data_dx_acorr_distributions_variance = variance_acorr(train_data_dx_acorr_distributions)
                    decoded_dx_acorr_distributions = autocorrelation_dx_distribution(decoded_dx,50)
                    decoded_dx_acorr_distributions_variance = variance_acorr(decoded_dx_acorr_distributions)
                    acorr_distributions_variance_difference_squared_mean = ((train_data_dx_acorr_distributions_variance - decoded_dx_acorr_distributions_variance)**2).mean()

                    # Batch velocity autocorrelation
                    train_data_dx_acorr_batch = autocorrelation_dx_torch(train_data_dx)
                    decoded_dx_acorr_batch = autocorrelation_dx_torch(decoded_dx)
                    acorr_batch_mean_difference_squared = ((decoded_dx_acorr_batch.mean(axis=0)-train_data_dx_acorr_batch.mean(axis=0))**2).mean()

                    # Mean jump
                    mean_jump_difference_squared_mean = mean_jump_loss(train_data_dx,decoded_dx)

                    # Reconstruction loss
                    mse = ((decoded_dx - train_data_dx)**2).mean()

                    # Weighting factors of each loss term
                    kurt_factor = 0.006
                    skew_factor = 1
                    var_factor = 500
                    mean_factor = 100000
                    acorr_factor = 100
                    mse_factor = 3
                    kl_factor = 0.01
                    acorr_batch_factor = 1500
                    mean_jump_factor = 100
                    acorr_distributions_tau1_percentiles_factor = 10
                    
                    # Loss function
                    loss = mse_factor*mse + kl_factor*kl_div + kurt_factor*kurtosis_difference_squared_mean\
                        + skew_factor*skewness_difference_squared_mean + var_factor*variance_difference_squared_mean\
                        + mean_factor*mean_difference_squared_mean + acorr_factor*acorr_difference_squared_mean\
                        + acorr_batch_factor*acorr_batch_mean_difference_squared + mean_jump_factor*mean_jump_difference_squared_mean\
                        + acorr_distributions_tau1_percentiles_factor*acorr_distributions_variance_difference_squared_mean

                    # Update losses
                    val_loss_epoch += loss.item() * len(data)
                    loss_kl_val += kl_div.item() * len(data)
                    loss_mse_val += mse.item() * len(data)
                    loss_kurtosis_val += kurtosis_difference_squared_mean.item() * len(data)
                    loss_skewness_val += skewness_difference_squared_mean.item() * len(data)
                    loss_variance_val += variance_difference_squared_mean.item() * len(data)
                    loss_mean_val += mean_difference_squared_mean.item() * len(data)
                    loss_acorr_val += acorr_difference_squared_mean.item() * len(data)
                    loss_mean_jump_val += mean_jump_difference_squared_mean.item() * len(data)
                    loss_acorr_batch_val += acorr_batch_mean_difference_squared.item() * len(data)
                    loss_acorr_percentiles_val += acorr_distributions_variance_difference_squared_mean.item() * len(data)
                    valset_size += len(data)
                
                #Append losses to their arrays
                self.validation_loss_per_epoch = np.append(self.validation_loss_per_epoch, val_loss_epoch/valset_size)
                self.validation_kl_loss_per_epoch = np.append(self.validation_kl_loss_per_epoch, loss_kl_val/valset_size)
                self.validation_mse_loss_per_epoch = np.append(self.validation_mse_loss_per_epoch, loss_mse_val/valset_size)
                self.validation_kurtosis_loss_per_epoch = np.append(self.validation_kurtosis_loss_per_epoch, loss_kurtosis_val/valset_size)
                self.validation_skewness_loss_per_epoch = np.append(self.validation_skewness_loss_per_epoch, loss_skewness_val/valset_size)
                self.validation_variance_loss_per_epoch = np.append(self.validation_variance_loss_per_epoch, loss_variance_val/valset_size)
                self.validation_mean_loss_per_epoch = np.append(self.validation_mean_loss_per_epoch, loss_mean_val/valset_size)
                self.validation_acorr_loss_per_epoch = np.append(self.validation_acorr_loss_per_epoch, loss_acorr_val/valset_size)
                self.validation_mean_jump_loss_per_epoch = np.append(self.validation_mean_jump_loss_per_epoch, loss_mean_jump_val/valset_size)
                self.validation_acorr_batch_loss_per_epoch = np.append(self.validation_acorr_batch_loss_per_epoch, loss_acorr_batch_val/valset_size)
                self.validation_acorr_percentiles_loss_per_epoch = np.append(self.validation_acorr_percentiles_loss_per_epoch, loss_acorr_percentiles_val/valset_size)
               
            self.epochs += 1
            
            # Print losses during training
            print(
                '\nepoch [{}/{}], Training loss:{:.2f}, KL-div training loss:{:.1f}, MSE training loss:{:.2f}, '
                'Kurtosis training loss:{:.2f}, Skewness training loss (x10e3):{:.2f}, '
                'Var training loss (x10e5):{:.2f}, Mean training loss (x10e7):{:.2f}, acorr training loss (x10e3):{:.2f}, acorr batch training loss (x10e4):{:.2f}, '
                'mean_jump training loss (x10e4):{:.2f}, acorr percentile train loss (x10e2): {:.2f}'.format(
                    self.epochs,
                    total_epochs,
                    loss_epoch/dataset_size,
                    loss_kl_training/dataset_size,
                    loss_mse_training/dataset_size,
                    loss_kurtosis_training/dataset_size,
                    1000*loss_skewness_training/dataset_size,
                    100000*loss_variance_training/dataset_size,
                    10000000*loss_mean_training/dataset_size,
                    1000*loss_acorr_training/dataset_size,
                    10000*loss_acorr_batch_training/dataset_size,
                    10000*loss_mean_jump_training/dataset_size,
                    100*loss_acorr_percentiles_training/dataset_size

                ), end=''
            )
            if val_loader:
                print(
                    ', Validation loss:{:.2f}, KL-div validation loss:{:.1f}, MSE validation loss:{:.2f}, '
                    'Kurtosis validation loss:{:.2f}, Skewness validation loss (x10e3):{:.2f}, '
                    'Var validation loss (x10e5):{:.2f}, Mean validation loss (x10e7):{:.2f}, acorr validation loss (x10e3):{:.2f}, acorr batch validation loss (x10e4):{:.2f}, '
                    'mean_jump validation loss (x10e4):{:.2f}, acorr percentile train loss (x10e2): {:.2f}'.format(
                        self.epochs,
                        total_epochs,
                        loss_epoch/dataset_size,
                        loss_kl_val/dataset_size,
                        loss_mse_val/dataset_size,
                        loss_kurtosis_val/dataset_size,
                        1000*loss_skewness_val/dataset_size,
                        100000*loss_variance_val/dataset_size,
                        10000000*loss_mean_val/dataset_size,
                        1000*loss_acorr_val/dataset_size,
                        10000*loss_acorr_batch_val/dataset_size,
                        10000*loss_mean_jump_val/dataset_size,
                        100*loss_acorr_percentiles_val/dataset_size

                      ),  end=''
                )
 
        print('\nmodel training completed')
        

    def get_model(name):
        """ Prints the trajectory length and the size of the latent space of the model saved in 'name'
        """
        model_dict = torch.load(name, map_location=device)
        print('Trajectory Length: ' + str(model_dict['length trajectories']))
        print('Latent Space Size: ' + str(model_dict['latent space']))
        return([model_dict['length trajectories'], model_dict['latent space']])
    

    def get_number_of_parameters(self):
        """Prints the number of parameters of the model
        """
        return sum(p.numel() for p in self.parameters())
    
    def save_model(self, parameters, name):
        """ Saves a dictionary in disk with the model state, its attributes and any training specifications defined in parameters
        """
        save_dict = {'model': self.state_dict(), 
                     'length trajectories': self.length_trajectory,
                     'latent space': self.latent_space,
                     'epochs': self.epochs}
        save_dict.update(parameters)
        save_dict['Training loss'] = self.training_loss_per_epoch
        save_dict['Training MSE loss'] = self.training_mse_loss_per_epoch
        save_dict['Training KL loss'] = self.training_kl_loss_per_epoch
        save_dict['Training Kurtosis loss'] = self.training_kurtosis_loss_per_epoch
        save_dict['Training Skewness loss'] = self.training_skewness_loss_per_epoch
        save_dict['Training Variance loss'] = self.training_variance_loss_per_epoch
        save_dict['Training Mean loss'] = self.training_mean_loss_per_epoch
        save_dict['Training Autocorrelation loss'] = self.training_acorr_loss_per_epoch
        save_dict['Training Mean Jump loss'] = self.training_mean_jump_loss_per_epoch
        save_dict['Training Autocorrelation batch loss'] = self.training_acorr_batch_loss_per_epoch
        save_dict['Training Autocorrelation percentiles loss'] = self.training_acorr_percentiles_loss_per_epoch

        save_dict['Validation loss'] = self.validation_loss_per_epoch
        save_dict['Validation MSE loss'] = self.validation_mse_loss_per_epoch
        save_dict['Validation KL loss'] = self.validation_kl_loss_per_epoch
        save_dict['Validation Kurtosis loss'] = self.validation_kurtosis_loss_per_epoch
        save_dict['Validation Skewness loss'] = self.validation_skewness_loss_per_epoch
        save_dict['Validation Variance loss'] = self.validation_variance_loss_per_epoch
        save_dict['Validation Mean loss'] = self.validation_mean_loss_per_epoch
        save_dict['Validation Autocorrelation loss'] = self.validation_acorr_loss_per_epoch
        save_dict['Validation Mean Jump loss'] = self.validation_mean_jump_loss_per_epoch
        save_dict['Validation Autocorrelation batch loss'] = self.validation_acorr_batch_loss_per_epoch
        save_dict['Validation Autocorrelation percentiles loss'] = self.validation_acorr_percentiles_loss_per_epoch

        torch.save(save_dict, name)
        print('model saved')
    
    def load_model(self, name, cuda=False, evaluate=False):
        """ Loads a trained model and a dictionary with all the saved training parameters
        """
        device = torch.device(device if cuda and torch.cuda.is_available() else 'cpu')
        self.model_dict = torch.load(name, map_location = device)
        self.load_state_dict(self.model_dict['model'])
        self.training_loss_per_epoch = self.model_dict['Training loss']
        self.training_mse_loss_per_epoch = self.model_dict['Training MSE loss']
        self.training_kl_loss_per_epoch = self.model_dict['Training KL loss']
        self.training_kurtosis_loss_per_epoch = self.model_dict['Training Kurtosis loss']
        self.training_skewness_loss_per_epoch = self.model_dict['Training Skewness loss']
        self.training_variance_loss_per_epoch = self.model_dict['Training Variance loss']
        self.training_mean_loss_per_epoch = self.model_dict['Training Mean loss']
        self.training_acorr_loss_per_epoch = self.model_dict['Training Autocorrelation loss']
        self.training_mean_jump_loss_per_epoch = self.model_dict['Training Mean Jump loss']
        self.training_acorr_batch_loss_per_epoch = self.model_dict['Training Autocorrelation batch loss']
        self.training_acorr_percentiles_loss_per_epoch = self.model_dict['Training Autocorrelation percentiles loss']

        self.validation_loss_per_epoch = self.model_dict['Validation loss']
        self.validation_mse_loss_per_epoch = self.model_dict['Validation MSE loss']
        self.validation_kl_loss_per_epoch = self.model_dict['Validation KL loss']
        self.validation_kurtosis_loss_per_epoch = self.model_dict['Validation Kurtosis loss']
        self.validation_skewness_loss_per_epoch = self.model_dict['Validation Skewness loss']
        self.validation_variance_loss_per_epoch = self.model_dict['Validation Variance loss']
        self.validation_mean_loss_per_epoch = self.model_dict['Validation Mean loss']
        self.validation_acorr_loss_per_epoch = self.model_dict['Validation Autocorrelation loss']
        self.validation_mean_jump_loss_per_epoch = self.model_dict['Validation Mean Jump loss']
        self.validation_acorr_batch_loss_per_epoch = self.model_dict['Validation Autocorrelation batch loss']
        self.validation_acorr_percentiles_loss_per_epoch = self.model_dict['Validation Autocorrelation percentiles loss']

        self.epochs = self.model_dict['epochs']
        if evaluate:
            self.eval()
        print('model loaded')
    
    def plot_loss(self, figsize=(36, 24), logscale=True, split='train'):
        """ Plots the losses of a trained model """
        if not self.training_loss_per_epoch.any():
            print('The model is not trained')
        else:
            plt.figure(figsize=figsize)
            if split == 'train':
                losses = [
                    self.training_loss_per_epoch,
                    self.training_mse_loss_per_epoch,
                    self.training_kl_loss_per_epoch,
                    self.training_kurtosis_loss_per_epoch,
                    self.training_skewness_loss_per_epoch,
                    self.training_variance_loss_per_epoch,
                    self.training_mean_loss_per_epoch,
                    self.training_acorr_loss_per_epoch,
                    self.training_mean_jump_loss_per_epoch,
                    self.training_acorr_batch_loss_per_epoch,
                    self.training_acorr_percentiles_loss_per_epoch
                ]
                titles = [
                    'Training Loss', 'Training MSE Loss', 'Training KL Loss', 'Training Kurtosis Loss',
                    'Training Skewness Loss', 'Training Variance Loss', 'Training Mean Loss',
                    'Training Acorr Loss', 'Training Mean Jump Loss',
                    'Training Acorr Batch Loss', 'Training Acorr Percentiles Loss'
                ]
            elif split == 'validation':
                losses = [
                    self.validation_loss_per_epoch,
                    self.validation_mse_loss_per_epoch.squeeze(),
                    self.validation_kl_loss_per_epoch.squeeze(),
                    self.validation_kurtosis_loss_per_epoch.squeeze(),
                    self.validation_skewness_loss_per_epoch.squeeze(),
                    self.validation_variance_loss_per_epoch.squeeze(),
                    self.validation_mean_loss_per_epoch.squeeze(),
                    self.validation_acorr_loss_per_epoch.squeeze(),
                    self.validation_mean_jump_loss_per_epoch.squeeze(),
                    self.validation_acorr_batch_loss_per_epoch.squeeze(),
                    self.validation_acorr_percentiles_loss_per_epoch.squeeze()
                ]
                titles = [
                    'Validation Loss', 'Validation MSE Loss', 'Validation KL Loss', 'Validation Kurtosis Loss',
                    'Validation Skewness Loss', 'Validation Variance Loss', 'Validation Mean Loss',
                    'Validation Acorr Loss', 'Validation Mean Jump Loss',
                    'Validation Acorr Batch Loss', 'Validation Acorr Percentiles Loss'
                ]
            elif split == 'both':
                training_losses = [
                    self.training_loss_per_epoch,
                    self.training_mse_loss_per_epoch,
                    self.training_kl_loss_per_epoch,
                    self.training_kurtosis_loss_per_epoch,
                    self.training_skewness_loss_per_epoch,
                    self.training_variance_loss_per_epoch,
                    self.training_mean_loss_per_epoch,
                    self.training_acorr_loss_per_epoch,
                    self.training_mean_jump_loss_per_epoch,
                    self.training_acorr_batch_loss_per_epoch,
                    self.training_acorr_percentiles_loss_per_epoch
                ]
                validation_losses = [
                    self.validation_loss_per_epoch,
                    self.validation_mse_loss_per_epoch.squeeze(),
                    self.validation_kl_loss_per_epoch.squeeze(),
                    self.validation_kurtosis_loss_per_epoch.squeeze(),
                    self.validation_skewness_loss_per_epoch.squeeze(),
                    self.validation_variance_loss_per_epoch.squeeze(),
                    self.validation_mean_loss_per_epoch.squeeze(),
                    self.validation_acorr_loss_per_epoch.squeeze(),
                    self.validation_mean_jump_loss_per_epoch.squeeze(),
                    self.validation_acorr_batch_loss_per_epoch.squeeze(),
                    self.validation_acorr_percentiles_loss_per_epoch.squeeze()
                ]
                titles = [
                    'Loss', 'MSE Loss', 'KL Loss', 'Kurtosis Loss',
                    'Skewness Loss', 'Variance Loss', 'Mean Loss',
                    'Acorr Loss', 'Mean Jump Loss',
                    'Acorr Batch Loss', 'Acorr Percentiles Loss'
                ]
            else:
                print('Choose split="train", "validation", or "both"')
                return

            fig, axs = plt.subplots(4, 3, figsize=figsize)
            fig.delaxes(axs[3][2])

            for i in range(11):
                row = i // 3
                col = i % 3
                if split == 'both':
                    if logscale:
                        axs[row][col].loglog(validation_losses[i], c='b', label='Validation')
                        axs[row][col].loglog(training_losses[i], c='r', label='Training')
                    else:
                        axs[row][col].plot(validation_losses[i], c='b', label='Validation')
                        axs[row][col].plot(training_losses[i], c='r', label='Training')
                    if i == 0:
                        axs[row][col].legend()
                else:
                    if logscale:
                        axs[row][col].loglog(losses[i], c='r')
                    else:
                        axs[row][col].plot(losses[i], c='r')
                axs[row][col].set_title(titles[i])


    
    def show_model_properties(self):
        """ Shows the training hyperparameters and any other stored model specifications
        """
        if self.training_loss_per_epoch.any():
            excludes = ['model', 'training loss', 'validation loss', 'optimizer state dict']
            keys = set(self.model_dict.keys())
            print('\033[1m'+'Training hyperparameters & diffusion models specifications:'+'\033[0m')
            for key in keys.difference(excludes):
                print(f'{key}: ' + str(self.model_dict[key]))
            print('Minimum training loss: ' + str(np.min(self.training_loss_per_epoch)))
            if self.validation_loss_per_epoch.any():
                print('Minumum validation loss: ' + str(np.min(self.validation_loss_per_epoch)))
                print('Best epoch (validation set): ' + str(np.argmin(self.validation_loss_per_epoch) + 1))
        else: print ('The model is not trained, there are no hyperparameters to show')


class SelfAttention(LEONARDO):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__(length_trajectory=200,latent_space=12)

        """ An object for calculating self attention in the encoder and decoder of the model
        """
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size) # Embed size needs to be divisble by heads

        self.values = nn.Linear(self.embed_size, self.embed_size, bias = False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias = False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias = False)
        self.fc_out = nn.Linear(heads*self.head_dim, self.embed_size)

    def forward(self, values, keys, queries):
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        N = queries.shape[0]
        value_len, key_len, queries_len = values.shape[1], keys.shape[1], queries.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, queries_len, self.heads, self.head_dim)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) #multiplying across the head_dim

        attention = torch.softmax(energy / (self.embed_size**0.5), dim=3) 
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, queries_len, self.heads*self.head_dim) 
        out = self.fc_out(out)

        return out
    

class TransformerBlock(LEONARDO):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__(length_trajectory=200,latent_space=12)

        """ An object for Transformer blocks, which exists inside the encoder and decoder
        """

        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size,embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, queries):
        attention = self.attention(values, keys, queries)
        x = self.dropout(self.norm1(attention + queries))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out
    

class Encoder(LEONARDO):
    def __init__(
            self,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            ):
        super(Encoder, self).__init__(length_trajectory=200,latent_space=12)

        """ An object for the encoder block
        """

        self.embed_size = embed_size
        self.device = device
        self.d_latent = self.latent_space
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                embed_size,
                heads,
                dropout=dropout,
                forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(1,embed_size)
        self.bottleneck = Bottleneck_Encoding(embed_size,self.length_trajectory)
        self.z_means, self.z_var = nn.Linear(512, self.d_latent), nn.Linear(512, self.d_latent)
    
    def reparameterize(self, mu, logvar, eps_scale=1):

        """Stochastic reparameterization
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) * eps_scale
        return mu + eps*std
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.linear1(x)
        out  = x

        for layerid,layer in enumerate(self.layers):
            out = layer(out,out,out)
        mem = out
        mem = mem.permute(0,2,1)
        mem = self.bottleneck(mem)
        mem = mem.contiguous().view(mem.size(0), -1)
        mu, logvar = self.z_means(mem), self.z_var(mem)
        mem = self.reparameterize(mu, logvar)
        return mem,mu,logvar
    
class DecoderBlock(LEONARDO): 
    def __init__(self, 
                 embed_size, 
                 heads, 
                 forward_expansion, 
                 dropout,
                 device):
        super(DecoderBlock,self).__init__(length_trajectory=200,latent_space=12)

        """An object for a decoder block
        """

        self.attention = SelfAttention(embed_size, heads=heads)
        self.norm = nn.LayerNorm(embed_size) 
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key):
        attention = self.attention(x, x, x)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query)
        return out


class Decoder(LEONARDO):
    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device
    ):
        super(Decoder, self).__init__(length_trajectory=200,latent_space=12)

        """An object for the decoder
        """

        self.device = device
        self.d_latent = self.latent_space
        self.embed_size = embed_size
        
        self.linear0 = nn.Linear(1,embed_size)
        self.linear1 = nn.Linear(self.d_latent,1024)
        self.bottleneck_decoding = Bottleneck_Decoding(embed_size,self.length_trajectory)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size,1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_out):
        enc_out = F.relu(self.linear1(enc_out))
        enc_out = self.bottleneck_decoding(enc_out)
        enc_out = enc_out.permute(0, 2, 1)
                
        for layerid,layer in enumerate(self.layers):
            x = layer(enc_out, enc_out, enc_out) 

        out = self.fc_out(x)
        return out


class TransformerVAE(LEONARDO):
    def __init__(
        self,
        embed_size=128,
        num_layers=2,
        forward_expansion=2,
        heads=8,
        dropout=0,
        device=device,
        length_trajectory=200,
        latent_space=12
    ):  
        super(TransformerVAE, self).__init__(length_trajectory=200,latent_space=12)

        """An object for the Transformer VAE model
        """

        self.encoder = Encoder(
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout
        )

        self.decoder = Decoder(
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
        )

        self.device = device

    def forward(self, src):
        enc_src,mu,logvar = self.encoder(src)
        
        out = self.decoder(enc_src)
        out = out.permute(0,2,1)
        return enc_src, mu, logvar, out



class Bottleneck_Encoding(LEONARDO):
    """
    Set of convolutional and dense layers to reduce input to a single
    latent vector
    """
    def __init__(self, size, timesteps):
        super().__init__(length_trajectory=200,latent_space=12)
        in_d = size
        timesteps = timesteps
        self.conv1 = nn.Conv1d(in_d,int(in_d/4), kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(int(in_d/4),int(in_d/4),kernel_size=2,stride=2)
        self.linear1 = nn.Linear(int((in_d/4)*(timesteps/2)),512)


    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x,1)
        x = torch.relu(self.linear1(x))
        return x
    

class Bottleneck_Decoding(LEONARDO):
    """
    Set of convolutional and dense layers to upsample latent vector to output
    that goes into the decoder blocks
    """
    def __init__(self, size, timesteps):
        super().__init__(length_trajectory=200,latent_space=12)
        self.in_d = size
        self.timesteps = timesteps
        self.linear1 = nn.Linear(1024,int((self.in_d/4)*(timesteps)))
        self.convtranspose1 = nn.ConvTranspose1d(int(self.in_d/4),int(self.in_d),kernel_size=7,padding=3)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = x.contiguous().view(-1,int(self.in_d/4),int(self.timesteps))
        x = self.convtranspose1(x)
        return x

    
    

