import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
import os
import math
from LEONARDO_utils import kurtosis_torch, skewness_torch, variance_torch, mean_torch, median_torch, autocorrelation_dx_torch, \
                 moments_loss, calculate_correlation_torch, autocorrelation_pos_torch \
            

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


class Leonardo(nn.Module):
    def __init__(self, length_trajectory, latent_space):
        super(Leonardo, self).__init__()

        """
        Initialize the Leonardo object, which will be used in the TransformerVAE model.

        # Args:
            length_trajectory (int): The length of the input trajectories.
            latent_space (int): The size of the latent space.

        Attributes:
            training_loss_per_epoch: Numpy arrays storing different loss components for training and validation.
            validation_loss_per_epoch: Arrays storing the validation losses at each epoch.
            epochs (int): Number of epochs for which the model has been trained.
        """
        self.length_trajectory = length_trajectory
        self.latent_space = latent_space
        
        # Initialize arrays to store losses over epochs
        self.training_loss_per_epoch = np.array([])
        self.training_kl_loss_per_epoch = np.array([])
        self.training_mse_loss_per_epoch = np.array([])
        self.training_kurtosis_loss_per_epoch = np.array([])
        self.training_skewness_loss_per_epoch = np.array([])
        self.training_variance_loss_per_epoch = np.array([])
        self.training_mean_loss_per_epoch = np.array([])
        self.training_median_loss_per_epoch = np.array([])
        self.training_acorr_loss_per_epoch = np.array([])
        self.training_acorr_batch_loss_per_epoch = np.array([])
        self.training_xycorr_loss_per_epoch = np.array([])
        self.training_pos_acorr_loss_per_epoch = np.array([])



        self.validation_loss_per_epoch = np.array([])
        self.validation_kl_loss_per_epoch = np.array([])
        self.validation_mse_loss_per_epoch = np.array([])
        self.validation_kurtosis_loss_per_epoch = np.array([])
        self.validation_skewness_loss_per_epoch = np.array([])
        self.validation_variance_loss_per_epoch = np.array([])
        self.validation_mean_loss_per_epoch = np.array([])
        self.validation_median_loss_per_epoch = np.array([])
        self.validation_acorr_loss_per_epoch = np.array([])
        self.validation_mean_jump_loss_per_epoch = np.array([])
        self.validation_acorr_batch_loss_per_epoch = np.array([])
        self.validation_acorr_percentiles_loss_per_epoch = np.array([])
        self.validation_xycorr_loss_per_epoch = np.array([])
        self.validation_pos_acorr_loss_per_epoch = np.array([])



        self.loss_weights = {
            'kurt_factor': 0.06,
            'skew_factor': 10,
            'var_factor': 500,
            'mean_factor': 50000,
            'median_factor': 1,
            'acorr_factor': 1000,
            'mse_factor': 0.001,
            'kl_factor': 0.05,
            'acorr_batch_factor': 100,
            'xycorr_factor': 10,
            'pos_acorr_factor': 1

        }

        # Initialize epoch count
        self.epochs = 0
    
    
    def train_Leonardo(self, epochs, dataloader, optimizer, device, val_loader=None):
        """
        Train the Leonardo model, updating training and validation losses over epochs.

        Args:
            epochs (int): Number of epochs to train for.
            dataloader (DataLoader): PyTorch dataloader for training data.
            optimizer (Optimizer): Optimizer for training.
            device (str): Device to run training on ('cpu' or 'cuda').
            val_loader (DataLoader, optional): PyTorch dataloader for validation data.

        Returns:
            None
        """

        total_epochs = self.epochs + epochs

        for epoch in range(epochs):
            # Train data
            loss_epoch, loss_kl_training, loss_mse_training, loss_kurtosis_training, loss_skewness_training, \
                loss_variance_training, loss_mean_training, loss_acorr_training, \
                    loss_acorr_batch_training, loss_xycorr_training, loss_pos_acorr_training, loss_median_training, \
                                dataset_size = (0,0,0,0,0,0,0,0,0,0,0,0,0)
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
                kurtosis_difference_squared_mean = moments_loss(train_data_dx,decoded_dx,kurtosis_torch)
                skewness_difference_squared_mean = moments_loss(train_data_dx,decoded_dx,skewness_torch)
                variance_difference_squared_mean = moments_loss(train_data_dx,decoded_dx,variance_torch)
                mean_difference_squared_mean = moments_loss(train_data_dx,decoded_dx,mean_torch)
                median_difference_squared_mean = moments_loss(train_data_dx,decoded_dx,median_torch)
                
                # Velocity autocorrelation mean
                # Define the weights for tau = 1 to 50
                tau_weights = 1 / torch.arange(1, 51, dtype=torch.float32, device=device)  # Shape: (50,) 
                train_data_dx_acorr = autocorrelation_dx_torch(train_data)[:,1:51]
                decoded_dx_acorr = autocorrelation_dx_torch(decoded)[:,1:51]
                acorr_difference_squared = (decoded_dx_acorr-train_data_dx_acorr)**2
                acorr_difference_squared_weighted = acorr_difference_squared*(tau_weights.view(1,-1,1))
                acorr_difference_squared_mean = acorr_difference_squared_weighted.mean()

                # Position autocorrelation
                train_data_pos_acorr = autocorrelation_pos_torch(train_data)
                decoded_pos_acorr = autocorrelation_pos_torch(decoded)
                acorr_pos_difference_squared_mean = ((decoded_pos_acorr-train_data_pos_acorr)**2).mean()

                # Batch velocity autocorrelation
                train_data_dx_acorr_batch = autocorrelation_dx_torch(train_data)
                decoded_dx_acorr_batch = autocorrelation_dx_torch(decoded)
                acorr_batch_mean_difference_squared = ((decoded_dx_acorr_batch.mean(axis=0)-train_data_dx_acorr_batch.mean(axis=0))**2).mean()

                # xy correlation 
                train_data_dx_xycorr = calculate_correlation_torch(train_data_dx)
                decoded_dx_xycorr = calculate_correlation_torch(decoded_dx)
                xycorr_difference_squared_mean = ((train_data_dx_xycorr-decoded_dx_xycorr)**2).mean()

                # Reconstruction loss
                mse = ((decoded_dx - train_data_dx)**2).mean()



                loss = (
                    self.loss_weights['mse_factor'] * mse +
                    self.loss_weights['kl_factor'] * kl_div +
                    self.loss_weights['kurt_factor'] * kurtosis_difference_squared_mean +
                    self.loss_weights['skew_factor'] * skewness_difference_squared_mean +
                    self.loss_weights['var_factor'] * variance_difference_squared_mean +
                    self.loss_weights['mean_factor'] * mean_difference_squared_mean +
                    self.loss_weights['median_factor'] * median_difference_squared_mean +
                    self.loss_weights['acorr_factor'] * acorr_difference_squared_mean +
                    self.loss_weights['acorr_batch_factor'] * acorr_batch_mean_difference_squared +
                    self.loss_weights['xycorr_factor'] * xycorr_difference_squared_mean + 
                    self.loss_weights['pos_acorr_factor'] * acorr_pos_difference_squared_mean
                )
                      


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
                loss_median_training += median_difference_squared_mean.item() * len(data)
                loss_acorr_training += acorr_difference_squared_mean.item() * len(data)
                loss_acorr_batch_training += acorr_batch_mean_difference_squared.item() * len(data)
                loss_xycorr_training += xycorr_difference_squared_mean.item() * len(data)
                loss_pos_acorr_training += acorr_pos_difference_squared_mean.item() * len(data)
                dataset_size += len(data)
            
            #Append losses to their arrays
            self.training_loss_per_epoch = np.append(self.training_loss_per_epoch, loss_epoch/dataset_size)
            self.training_kl_loss_per_epoch = np.append(self.training_kl_loss_per_epoch, loss_kl_training/dataset_size)
            self.training_mse_loss_per_epoch = np.append(self.training_mse_loss_per_epoch, loss_mse_training/dataset_size)
            self.training_kurtosis_loss_per_epoch = np.append(self.training_kurtosis_loss_per_epoch, loss_kurtosis_training/dataset_size)
            self.training_skewness_loss_per_epoch = np.append(self.training_skewness_loss_per_epoch, loss_skewness_training/dataset_size)
            self.training_variance_loss_per_epoch = np.append(self.training_variance_loss_per_epoch, loss_variance_training/dataset_size)
            self.training_mean_loss_per_epoch = np.append(self.training_mean_loss_per_epoch, loss_mean_training/dataset_size)
            self.training_median_loss_per_epoch = np.append(self.training_median_loss_per_epoch, loss_median_training/dataset_size)
            self.training_acorr_loss_per_epoch = np.append(self.training_acorr_loss_per_epoch, loss_acorr_training/dataset_size)
            self.training_acorr_batch_loss_per_epoch = np.append(self.training_acorr_batch_loss_per_epoch, loss_acorr_batch_training/dataset_size)
            self.training_xycorr_loss_per_epoch = np.append(self.training_xycorr_loss_per_epoch, loss_xycorr_training/dataset_size)
            self.training_pos_acorr_loss_per_epoch = np.append(self.training_pos_acorr_loss_per_epoch, loss_pos_acorr_training/dataset_size)


            ## Validation
            if val_loader:
                val_loss_epoch, loss_kl_val, loss_mse_val, loss_kurtosis_val, loss_skewness_val, \
                    loss_variance_val, loss_mean_val, loss_acorr_val, \
                        loss_acorr_batch_val, \
                            loss_xycorr_val, loss_pos_acorr_val, loss_median_val, \
                                valset_size = (0,0,0,0,0,0,0,0,0,0,0,0,0,)
                for data in val_loader:
                    val_data = Variable(data).to(device)
                    val_data_dx = val_data.squeeze()[:,1:]-val_data.squeeze()[:,0:-1]

                    _, z_mean, z_log_var, decoded = self(val_data)
                    decoded_dx = decoded.squeeze()[:,1:]-decoded.squeeze()[:,0:-1]
                 
                    kl_div = -0.5*torch.sum(1 + z_log_var 
                                            - z_mean**2 
                                            - torch.exp(z_log_var),
                                            axis = 1
                                            )
                    kl_div = kl_div.mean()

                    ## Calculating loss function components:

                    # Four moments of the distribution of displacements
                    kurtosis_difference_squared_mean = moments_loss(val_data_dx,decoded_dx,kurtosis_torch)
                    skewness_difference_squared_mean = moments_loss(val_data_dx,decoded_dx,skewness_torch)
                    variance_difference_squared_mean = moments_loss(val_data_dx,decoded_dx,variance_torch)
                    mean_difference_squared_mean = moments_loss(val_data_dx,decoded_dx,mean_torch)
                    median_difference_squared_mean = moments_loss(val_data_dx,decoded_dx,median_torch)
                    
                    # Velocity autocorrelation mean
                    # Define the weights for tau = 1 to 50
                    tau_weights = 1 / torch.arange(1, 51, dtype=torch.float32, device=device)  # Shape: (50,) 
                    val_data_dx_acorr = autocorrelation_dx_torch(val_data)[:,1:51]
                    decoded_dx_acorr = autocorrelation_dx_torch(decoded)[:,1:51]
                    acorr_difference_squared = (decoded_dx_acorr-val_data_dx_acorr)**2
                    acorr_difference_squared_weighted = acorr_difference_squared*(tau_weights.view(1,-1,1))
                    acorr_difference_squared_mean = acorr_difference_squared_weighted.mean()

                    # Position autocorrelation
                    val_data_pos_acorr = autocorrelation_pos_torch(val_data)
                    decoded_pos_acorr = autocorrelation_pos_torch(decoded)
                    acorr_pos_difference_squared_mean = ((decoded_pos_acorr-val_data_pos_acorr)**2).mean()

                    # Batch velocity autocorrelation
                    val_data_dx_acorr_batch = autocorrelation_dx_torch(val_data)
                    decoded_dx_acorr_batch = autocorrelation_dx_torch(decoded)
                    acorr_batch_mean_difference_squared = ((decoded_dx_acorr_batch.mean(axis=0)-val_data_dx_acorr_batch.mean(axis=0))**2).mean()

                    # xy correlation 
                    val_data_dx_asymm = calculate_correlation_torch(val_data_dx)
                    decoded_dx_asymm = calculate_correlation_torch(decoded_dx)
                    xycorr_difference_squared_mean = ((val_data_dx_asymm-decoded_dx_asymm)**2).mean()

                    # Reconstruction loss
                    mse = ((decoded_dx - val_data_dx)**2).mean()



                    loss = (
                        self.loss_weights['mse_factor'] * mse +
                        self.loss_weights['kl_factor'] * kl_div +
                        self.loss_weights['kurt_factor'] * kurtosis_difference_squared_mean +
                        self.loss_weights['skew_factor'] * skewness_difference_squared_mean +
                        self.loss_weights['var_factor'] * variance_difference_squared_mean +
                        self.loss_weights['mean_factor'] * mean_difference_squared_mean +
                        self.loss_weights['median_factor'] * median_difference_squared_mean +
                        self.loss_weights['acorr_factor'] * acorr_difference_squared_mean +
                        self.loss_weights['acorr_batch_factor'] * acorr_batch_mean_difference_squared +
                        self.loss_weights['xycorr_factor'] * xycorr_difference_squared_mean + 
                        self.loss_weights['pos_acorr_factor'] * acorr_pos_difference_squared_mean
                    )

            


                    # Update losses
                    val_loss_epoch += loss.item() * len(data)
                    loss_kl_val += kl_div.item() * len(data)
                    loss_mse_val += mse.item() * len(data)
                    loss_kurtosis_val += kurtosis_difference_squared_mean.item() * len(data)
                    loss_skewness_val += skewness_difference_squared_mean.item() * len(data)
                    loss_variance_val += variance_difference_squared_mean.item() * len(data)
                    loss_mean_val += mean_difference_squared_mean.item() * len(data)
                    loss_median_val += median_difference_squared_mean.item() * len(data)
                    loss_acorr_val += acorr_difference_squared_mean.item() * len(data)
                    loss_acorr_batch_val += acorr_batch_mean_difference_squared.item() * len(data)
                    loss_xycorr_val += xycorr_difference_squared_mean.item() * len(data)
                    loss_pos_acorr_val += acorr_pos_difference_squared_mean.item() * len(data)
                    valset_size += len(data)
                
                #Append losses to their arrays
                self.validation_loss_per_epoch = np.append(self.validation_loss_per_epoch, val_loss_epoch/valset_size)
                self.validation_kl_loss_per_epoch = np.append(self.validation_kl_loss_per_epoch, loss_kl_val/valset_size)
                self.validation_mse_loss_per_epoch = np.append(self.validation_mse_loss_per_epoch, loss_mse_val/valset_size)
                self.validation_kurtosis_loss_per_epoch = np.append(self.validation_kurtosis_loss_per_epoch, loss_kurtosis_val/valset_size)
                self.validation_skewness_loss_per_epoch = np.append(self.validation_skewness_loss_per_epoch, loss_skewness_val/valset_size)
                self.validation_variance_loss_per_epoch = np.append(self.validation_variance_loss_per_epoch, loss_variance_val/valset_size)
                self.validation_mean_loss_per_epoch = np.append(self.validation_mean_loss_per_epoch, loss_mean_val/valset_size)
                self.validation_median_loss_per_epoch = np.append(self.validation_median_loss_per_epoch, loss_median_val/valset_size)
                self.validation_acorr_loss_per_epoch = np.append(self.validation_acorr_loss_per_epoch, loss_acorr_val/valset_size)
                self.validation_acorr_batch_loss_per_epoch = np.append(self.validation_acorr_batch_loss_per_epoch, loss_acorr_batch_val/valset_size)
                self.validation_xycorr_loss_per_epoch = np.append(self.validation_xycorr_loss_per_epoch, loss_xycorr_val/valset_size)
                self.validation_pos_acorr_loss_per_epoch = np.append(self.validation_pos_acorr_loss_per_epoch, loss_pos_acorr_val/valset_size)              
            self.epochs += 1
            
            # Print losses during training
            print(
                '\nepoch [{}/{}], Training loss:{:.2f}, KL-div training loss:{:.1f}, MSE training loss:{:.2f}, '
                'Kurtosis training loss:{:.2f}, Skewness training loss (x10e3):{:.2f}, '
                'Var training loss (x10e5):{:.2f}, Mean training loss (x10e7):{:.2f}, Median training loss (x10e7):{:.2f},' 
                'acorr training loss (x10e3):{:.2f}, acorr batch training loss (x10e4):{:.2f}, '
                'xycorr factor training loss (x10e3): {:.3f},'
                  'pos acorr training loss (x10e3): {:.3f}\n'.format(
                    self.epochs,
                    total_epochs,
                    loss_epoch/dataset_size,
                    loss_kl_training/dataset_size,
                    loss_mse_training/dataset_size,
                    loss_kurtosis_training/dataset_size,
                    1000*loss_skewness_training/dataset_size,
                    100000*loss_variance_training/dataset_size,
                    10000000*loss_mean_training/dataset_size,
                    10000000*loss_median_training/dataset_size,
                    1000*loss_acorr_training/dataset_size,
                    10000*loss_acorr_batch_training/dataset_size,
                    1000*loss_xycorr_training/dataset_size,
                    1000*loss_pos_acorr_training/dataset_size

                ), end=''
            )
            if val_loader:
                print(
                    'Validation loss:{:.2f}, KL-div val loss:{:.1f}, MSE val loss:{:.2f}, '
                    'Kurtosis val loss:{:.2f}, Skewness val loss (x10e3):{:.2f}, '
                    'Var val loss (x10e5):{:.2f}, Mean val loss (x10e7):{:.2f}, Median val loss (x10e7):{:.2f}, '
                    'acorr val loss (x10e3):{:.2f}, acorr batch val loss (x10e4):{:.2f}, '
                    
                    'xycorr val loss (x10e3): {:.3f}, pos acorr val loss (x10e3): {:.3f}'.format(
                        val_loss_epoch/valset_size,
                        loss_kl_val/valset_size,
                        loss_mse_val/valset_size,
                        loss_kurtosis_val/valset_size,
                        1000*loss_skewness_val/valset_size,
                        100000*loss_variance_val/valset_size,
                        10000000*loss_mean_val/valset_size,
                        10000000*loss_median_val/valset_size,
                        1000*loss_acorr_val/valset_size,
                        10000*loss_acorr_batch_val/valset_size,
                        1000*loss_xycorr_val/valset_size,
                        1000*loss_pos_acorr_val/valset_size
                      ),  end=''
                )
 
        print('\nmodel training completed')
        

    def get_model(name):
        """ Prints the trajectory length and the size of the latent space of the model saved in 'name'
        """
        model_dict = torch.load(name, map_location=device)
        print('Trajectory Length: ' + str(model_dict['Trajectory Length']))
        print('Latent Space Size: ' + str(model_dict['Latent Space']))
        return([model_dict['Trajectory Length'], model_dict['Latent Space']])
    

    def get_number_of_parameters(self):
        """Prints the number of parameters of the model
        """
        return sum(p.numel() for p in self.parameters())
    
    def save_model(self, parameters, name):
        """ Saves a dictionary in disk with the model state, its attributes and any training specifications defined in parameters
        """
        save_dict = {'Model': self.state_dict(),
                     'Trajectory Length': self.length_trajectory,
                     'Latent Space': self.latent_space,
                     'Epochs': self.epochs,
                     'Training Loss': self.training_loss_per_epoch,
                     'Validation Loss': self.validation_loss_per_epoch,
                     'Model Architecture': {
                        'Transformer Architecture': {
                            'embed size': self.encoder.embed_size,  
                            'num layers': len(self.encoder.layers),  
                            'heads': self.encoder.layers[0].attention.heads, 
                            'forward expansion': self.encoder.layers[0].feed_forward[0].out_features // self.encoder.embed_size
                            },
                        'Bottleneck Architecture': {
                            'encoding': {
                                'conv1': {
                                    'in_channels': self.encoder.bottleneck.conv1.in_channels,
                                    'out_channels': self.encoder.bottleneck.conv1.out_channels,
                                    'kernel_size': self.encoder.bottleneck.conv1.kernel_size,
                                    'stride': self.encoder.bottleneck.conv1.stride,
                                    'padding': self.encoder.bottleneck.conv1.padding
                                },
                                'conv2': {
                                    'in_channels': self.encoder.bottleneck.conv2.in_channels,
                                    'out_channels': self.encoder.bottleneck.conv2.out_channels,
                                    'kernel_size': self.encoder.bottleneck.conv2.kernel_size,
                                    'stride': self.encoder.bottleneck.conv2.stride
                                },
                                'linear1': {
                                    'in_features': self.encoder.bottleneck.linear1.in_features,
                                    'out_features': self.encoder.bottleneck.linear1.out_features
                                }
                            },
                            'decoding': {
                                'linear1': {
                                    'in_features': self.decoder.bottleneck_decoding.linear1.in_features,
                                    'out_features': self.decoder.bottleneck_decoding.linear1.out_features
                                },
                                'convtranspose1': {
                                    'in_channels': self.decoder.bottleneck_decoding.convtranspose1.in_channels,
                                    'out_channels': self.decoder.bottleneck_decoding.convtranspose1.out_channels,
                                    'kernel_size': self.decoder.bottleneck_decoding.convtranspose1.kernel_size,
                                    'stride': self.decoder.bottleneck_decoding.convtranspose1.stride,
                                    'padding': self.decoder.bottleneck_decoding.convtranspose1.padding
                                    }
                                }
                            }
                        }
                    }
                        
        
        parameters['loss weights'] = self.loss_weights     
        save_dict.update(parameters)
        save_dict['Training loss'] = self.training_loss_per_epoch
        save_dict['Training MSE loss'] = self.training_mse_loss_per_epoch
        save_dict['Training KL loss'] = self.training_kl_loss_per_epoch
        save_dict['Training Kurtosis loss'] = self.training_kurtosis_loss_per_epoch
        save_dict['Training Skewness loss'] = self.training_skewness_loss_per_epoch
        save_dict['Training Variance loss'] = self.training_variance_loss_per_epoch
        save_dict['Training Mean loss'] = self.training_mean_loss_per_epoch
        save_dict['Training Median loss'] = self.training_median_loss_per_epoch
        save_dict['Training Autocorrelation loss'] = self.training_acorr_loss_per_epoch
        save_dict['Training Autocorrelation batch loss'] = self.training_acorr_batch_loss_per_epoch
        save_dict['Training xy-Correlation loss'] = self.training_xycorr_loss_per_epoch
        save_dict['Training Positional Acorr loss'] = self.training_pos_acorr_loss_per_epoch


        save_dict['Validation loss'] = self.validation_loss_per_epoch
        save_dict['Validation MSE loss'] = self.validation_mse_loss_per_epoch
        save_dict['Validation KL loss'] = self.validation_kl_loss_per_epoch
        save_dict['Validation Kurtosis loss'] = self.validation_kurtosis_loss_per_epoch
        save_dict['Validation Skewness loss'] = self.validation_skewness_loss_per_epoch
        save_dict['Validation Variance loss'] = self.validation_variance_loss_per_epoch
        save_dict['Validation Mean loss'] = self.validation_mean_loss_per_epoch
        save_dict['Validation Median loss'] = self.validation_median_loss_per_epoch
        save_dict['Validation Autocorrelation loss'] = self.validation_acorr_loss_per_epoch
        save_dict['Validation Autocorrelation batch loss'] = self.validation_acorr_batch_loss_per_epoch
        save_dict['Validation xy-Correlation loss'] = self.validation_xycorr_loss_per_epoch
        save_dict['Validation Positional Acorr loss'] = self.validation_pos_acorr_loss_per_epoch


        torch.save(save_dict, name)
        print('Model Saved')
    
    def load_model(self, name, cuda=False, evaluate=False):
        """ Loads a trained model and a dictionary with all the saved training parameters
        """
        device = torch.device(device if cuda and torch.cuda.is_available() else 'cpu')
        self.model_dict = torch.load(name, map_location = device)
        self.load_state_dict(self.model_dict['Model'])
        self.training_loss_per_epoch = self.model_dict['Training loss']
        self.training_mse_loss_per_epoch = self.model_dict['Training MSE loss']
        self.training_kl_loss_per_epoch = self.model_dict['Training KL loss']
        self.training_kurtosis_loss_per_epoch = self.model_dict['Training Kurtosis loss']
        self.training_skewness_loss_per_epoch = self.model_dict['Training Skewness loss']
        self.training_variance_loss_per_epoch = self.model_dict['Training Variance loss']
        self.training_mean_loss_per_epoch = self.model_dict['Training Mean loss']
        self.training_median_loss_per_epoch = self.model_dict['Training Median loss']
        self.training_acorr_loss_per_epoch = self.model_dict['Training Autocorrelation loss']
        self.training_acorr_batch_loss_per_epoch = self.model_dict['Training Autocorrelation batch loss']
        self.training_xycorr_loss_per_epoch = self.model_dict['Training xy-Correlation loss']
        self.training_pos_acorr_loss_per_epoch = self.model_dict['Training Positional Acorr loss']



        self.validation_loss_per_epoch = self.model_dict['Validation loss']
        self.validation_mse_loss_per_epoch = self.model_dict['Validation MSE loss']
        self.validation_kl_loss_per_epoch = self.model_dict['Validation KL loss']
        self.validation_kurtosis_loss_per_epoch = self.model_dict['Validation Kurtosis loss']
        self.validation_skewness_loss_per_epoch = self.model_dict['Validation Skewness loss']
        self.validation_variance_loss_per_epoch = self.model_dict['Validation Variance loss']
        self.validation_mean_loss_per_epoch = self.model_dict['Validation Mean loss']
        self.validation_median_loss_per_epoch = self.model_dict['Validation Median loss']
        self.validation_acorr_loss_per_epoch = self.model_dict['Validation Autocorrelation loss']
        self.validation_acorr_batch_loss_per_epoch = self.model_dict['Validation Autocorrelation batch loss']
        self.validation_xycorr_loss_per_epoch = self.model_dict['Validation xy-Correlation loss']
        self.validation_pos_acorr_loss_per_epoch = self.model_dict['Validation Positional Acorr loss']


        self.epochs = self.model_dict['Epochs']
        if evaluate:
            self.eval()
        print('Model Loaded')
    
    def plot_loss(self, figsize=(36, 15), logscale=True, split='train'):
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
                    self.training_median_loss_per_epoch,
                    self.training_acorr_loss_per_epoch,
                    self.training_mean_jump_loss_per_epoch,
                    self.training_acorr_batch_loss_per_epoch,
                    self.training_acorr_percentiles_loss_per_epoch,
                    self.training_xycorr_loss_per_epoch,
                    self.training_pos_acorr_loss_per_epoch,


                ]
                titles = [
                    'Training Loss', 'Training MSE Loss', 'Training KL Loss', 'Training Kurtosis Loss',
                    'Training Skewness Loss', 'Training Variance Loss', 'Training Mean Loss', 'Training Median Loss',
                    'Training Acorr Loss', 'Training Acorr Batch Loss', 'Training xy_Correlation Loss', 'Training Positional Acorr Loss', 
                ]
            elif split == 'validation':
                losses = [
                    self.validation_loss_per_epoch,
                    self.validation_mse_loss_per_epoch,
                    self.validation_kl_loss_per_epoch,
                    self.validation_kurtosis_loss_per_epoch,
                    self.validation_skewness_loss_per_epoch,
                    self.validation_variance_loss_per_epoch,
                    self.validation_mean_loss_per_epoch,
                    self.validation_median_loss_per_epoch,
                    self.validation_acorr_loss_per_epoch,
                    self.validation_acorr_batch_loss_per_epoch,
                    self.validation_xycorr_loss_per_epoch,
                    self.validation_pos_acorr_loss_per_epoch,

                ]
                titles = [
                    'Validation Loss', 'Validation MSE Loss', 'Validation KL Loss', 'Validation Kurtosis Loss',
                    'Validation Skewness Loss', 'Validation Variance Loss', 'Validation Mean Loss', 'Validation Median Loss',
                    'Validation Acorr Loss', 'Validation Acorr Batch Loss', 'Validation xy_Correlation Loss', 'Validation Positional Acorr Loss', 
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
                    self.training_median_loss_per_epoch,
                    self.training_acorr_loss_per_epoch,
                    self.training_acorr_batch_loss_per_epoch,
                    self.training_xycorr_loss_per_epoch,
                    self.training_pos_acorr_loss_per_epoch,

                ]
                validation_losses = [
                    self.validation_loss_per_epoch,
                    self.validation_mse_loss_per_epoch,
                    self.validation_kl_loss_per_epoch,
                    self.validation_kurtosis_loss_per_epoch,
                    self.validation_skewness_loss_per_epoch,
                    self.validation_variance_loss_per_epoch,
                    self.validation_mean_loss_per_epoch,
                    self.validation_median_loss_per_epoch,
                    self.validation_acorr_loss_per_epoch,
                    self.validation_acorr_batch_loss_per_epoch,
                    self.validation_xycorr_loss_per_epoch,
                    self.validation_pos_acorr_loss_per_epoch,

                ]
                titles = [
                    'Loss', 'MSE Loss', 'KL Loss', 'Kurtosis Loss',
                    'Skewness Loss', 'Variance Loss', 'Mean Loss', 'Median Loss',
                    'Acorr Loss', 'Acorr Batch Loss', 'xy_Correlation Loss', 'Positional Acorr Loss'
                ]
            else:
                print('Choose split="train", "validation", or "both"')
                return

            fig, axs = plt.subplots(3, 4, figsize=figsize)
            axs = axs.flatten()


            for i in range(12):
                # row = i // 6
                # col = i % 3
                if split == 'both':
                    if logscale:
                        axs[i].loglog(validation_losses[i], c='b', label='Validation')
                        axs[i].loglog(training_losses[i], c='r', label='Training')
                    else:
                        axs[i].plot(validation_losses[i], c='b', label='Validation')
                        axs[i].plot(training_losses[i], c='r', label='Training')
                    if i == 0:
                        axs[i].legend()
                else:
                    if logscale:
                        axs[i].loglog(losses[i], c='r')
                    else:
                        axs[i].plot(losses[i], c='r')
                axs[i].set_title(titles[i])


    
    def show_model_properties(self):
        """ Shows the training hyperparameters and any other stored model specifications
        """
        if self.training_loss_per_epoch.any():
            excludes = ['Model', 'Training Loss', 'Validation Loss', 'optimizer state dict']
            keys = set(self.model_dict.keys())
            print('\033[1m'+'Training hyperparameters other model parameters:'+'\033[0m')
            for key in keys.difference(excludes):
                print(f'{key}: ' + str(self.model_dict[key]))
            print('Minimum training loss: ' + str(np.min(self.training_loss_per_epoch)))
            if self.validation_loss_per_epoch.any():
                print('Minumum validation loss: ' + str(np.min(self.validation_loss_per_epoch)))
                print('Best epoch (validation set): ' + str(np.argmin(self.validation_loss_per_epoch) + 1))
        else: print ('The model is not trained, there are no hyperparameters to show')













## Model Architecture

class SelfAttention(Leonardo):
    def __init__(self, embed_size, heads):
        """
        Initializes the SelfAttention module.
        
        Args:
            embed_size (int): Dimensionality of input feature vectors.
            heads (int): Number of attention heads.
            
        The attention mechanism splits the input into multiple heads, performs attention, and then concatenates the results.
        """
        super(SelfAttention, self).__init__(length_trajectory=200, latent_space=12)
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        # Ensure the embedding size is divisible by the number of heads
        assert (self.head_dim * heads == embed_size)

        # Linear transformations for queries, keys, and values
        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, self.embed_size)

    def forward(self, values, keys, queries):
        """
        Forward pass of self-attention.
        
        Args:
            values (tensor): Input values (batch_size, sequence_length, embed_size).
            keys (tensor): Input keys (batch_size, sequence_length, embed_size).
            queries (tensor): Input queries (batch_size, sequence_length, embed_size).
            
        Returns:
            tensor: Output after applying the attention mechanism.
        """
        N = queries.shape[0]
        value_len, key_len, queries_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Linear transformation for values, keys, and queries
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Split the embedding into multiple heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, queries_len, self.heads, self.head_dim)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size**0.5), dim=3)

        # Combine the attention values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, queries_len, self.heads * self.head_dim)

        # Final linear transformation
        out = self.fc_out(out)

        return out


class TransformerBlock(Leonardo):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        """
        Initializes a Transformer block.
        
        Args:
            embed_size (int): Dimensionality of input feature vectors.
            heads (int): Number of attention heads.
            dropout (float): Dropout rate.
            forward_expansion (int): Expansion factor for the feed-forward layer.
            
        This block applies self-attention followed by a feed-forward neural network.
        """
        super(TransformerBlock, self).__init__(length_trajectory=200, latent_space=12)
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, queries):
        """
        Forward pass through the transformer block.
        
        Args:
            values (tensor): Input values (batch_size, sequence_length, embed_size).
            keys (tensor): Input keys (batch_size, sequence_length, embed_size).
            queries (tensor): Input queries (batch_size, sequence_length, embed_size).
            
        Returns:
            tensor: Output after attention and feed-forward layers.
        """
        # Self-attention followed by normalization and residual connection
        attention = self.attention(values, keys, queries)
        x = self.dropout(self.norm1(attention + queries))

        # Feed-forward network followed by normalization and residual connection
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out


class Encoder(Leonardo):
    def __init__(self, embed_size, num_layers, heads, device, forward_expansion, dropout):
        """
        Initializes the Encoder module.
        
        Args:
            embed_size (int): Dimensionality of input feature vectors.
            num_layers (int): Number of transformer layers in the encoder.
            heads (int): Number of attention heads in each layer.
            device (str): Device to use for training (e.g., 'cuda').
            forward_expansion (int): Expansion factor for the feed-forward layer.
            dropout (float): Dropout rate.
            
        This encoder is made up of multiple transformer blocks.
        """
        super(Encoder, self).__init__(length_trajectory=200, latent_space=12)
        self.embed_size = embed_size
        self.device = device
        self.d_latent = self.latent_space
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=embed_size, kernel_size=3, padding=1)  # Convert 2D trajectories to embed_size.
        self.bottleneck = Bottleneck_Encoding(embed_size, self.length_trajectory)
        self.z_means = nn.Linear(512, self.d_latent)
        self.z_var = nn.Linear(512, self.d_latent)

    def positional_encoding(self, sequence_length, embed_size):
        pe = torch.zeros(sequence_length, embed_size)
        for pos in range(sequence_length):
            for i in range(0, embed_size, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / embed_size)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * i / embed_size)))
        return pe.to(self.device)  # Ensure it’s on the same device as the model

    def reparameterize(self, mu, logvar, eps_scale=1):
        """
        Applies the reparameterization trick to create latent vectors.

        Args:
            mu (tensor): Mean of the latent distribution.
            logvar (tensor): Log variance of the latent distribution.
            eps_scale (float): Scaling factor for noise.

        Returns:
            tensor: Sampled latent vector using the reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * eps_scale
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (tensor): Input trajectories (batch_size, sequence_length, 2).

        Returns:
            tuple: Encoded latent vector, mean, and log variance.
        """
        
        x = x.permute(0, 2, 1) # Need to add this if using convolutional embedding instead of linear
        x = self.conv1(x)
        x = x.permute(0, 2, 1) # Need to add this if using convolutional embedding instead of linear

        out = x

        for layerid, layer in enumerate(self.layers):
            out = layer(out, out, out)

        mem = out.permute(0, 2, 1)
        mem = self.bottleneck(mem)
        mem = mem.contiguous().view(mem.size(0), -1)
        mu, logvar = self.z_means(mem), self.z_var(mem)
        mem = self.reparameterize(mu, logvar)

        return mem, mu, logvar

    
class DecoderBlock(Leonardo): 
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        """
        Initializes the Decoder block.
        
        Args:
            embed_size (int): Dimensionality of input feature vectors.
            heads (int): Number of attention heads.
            forward_expansion (int): Expansion factor for the feed-forward layer.
            dropout (float): Dropout rate.
            device (str): Device to use for training (e.g., 'cuda').
            
        This block applies self-attention followed by a transformer block for decoding.
        """
        super(DecoderBlock, self).__init__(length_trajectory=200, latent_space=12)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key):
        """
        Forward pass through the decoder block.
        
        Args:
            x (tensor): Input tensor (batch_size, sequence_length, embed_size).
            value (tensor): Value tensor for self-attention (batch_size, sequence_length, embed_size).
            key (tensor): Key tensor for self-attention (batch_size, sequence_length, embed_size).
            
        Returns:
            tensor: Output after self-attention and transformer block.
        """
        # Self-attention followed by a transformer block
        attention = self.attention(x, x, x)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query)
        return out


class Decoder(Leonardo):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout, device):
        """
        Initializes the Decoder module.
        
        Args:
            embed_size (int): Dimensionality of input feature vectors.
            num_layers (int): Number of transformer layers in the decoder.
            heads (int): Number of attention heads in each layer.
            forward_expansion (int): Expansion factor for the feed-forward layer.
            dropout (float): Dropout rate.
            device (str): Device to use for training (e.g., 'cuda').
            
        This decoder is made up of multiple decoder blocks.
        """
        super(Decoder, self).__init__(length_trajectory=200, latent_space=12)
        self.device = device
        self.d_latent = self.latent_space
        self.embed_size = embed_size

        self.linear1 = nn.Linear(self.d_latent, 1024)  # Maps latent space to bottleneck input size.
        self.bottleneck_decoding = Bottleneck_Decoding(embed_size, self.length_trajectory)
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)]
        )
        self.conv_out = nn.Conv1d(embed_size, 2, kernel_size=3, padding=1)  # Final output layer to map back to 2D trajectories.
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_out):
        """
        Forward pass through the decoder.
        
        Args:
            enc_out (tensor): Encoded latent vector (batch_size, latent_space_size).
            
        Returns:
            tensor: Decoded output trajectories (batch_size, sequence_length, 2).
        """
        # Decode the latent vector back to a trajectory representation
        enc_out = F.relu(self.linear1(enc_out))
        enc_out = self.bottleneck_decoding(enc_out)
        enc_out = enc_out.permute(0, 2, 1)

        # Pass through the decoder layers
        for layerid, layer in enumerate(self.layers):
            x = layer(enc_out, enc_out, enc_out)

        # Final output layer
        x = x.permute(0, 2, 1) # Need to add this if using convolutional layer instead of linear
        out = self.conv_out(x)
        out = out.permute(0, 2, 1) # Need to add this if using convolutional layer instead of linear

        return out


class TransformerVAE(Leonardo):
    def __init__(self, embed_size=128, num_layers=2, forward_expansion=2, heads=8, dropout=0.0, device=device, length_trajectory=200, latent_space=12):
        """
        Initializes the TransformerVAE model.
        
        Args:
            embed_size (int): Dimensionality of input feature vectors.
            num_layers (int): Number of transformer layers in both the encoder and decoder.
            forward_expansion (int): Expansion factor for the feed-forward layers.
            heads (int): Number of attention heads in each layer.
            dropout (float): Dropout rate.
            device (str): Device to use for training (e.g., 'cuda').
            length_trajectory (int): Length of input trajectories.
            latent_space (int): Size of the latent space.
            
        The TransformerVAE model combines an encoder and a decoder using a variational autoencoder architecture.
        """
        super(TransformerVAE, self).__init__(length_trajectory=length_trajectory, latent_space=latent_space)
        self.encoder = Encoder(embed_size, num_layers, heads, device, forward_expansion, dropout)
        self.decoder = Decoder(embed_size, num_layers, heads, forward_expansion, dropout, device)
        self.device = device
        self.layernorm = nn.LayerNorm(length_trajectory)

    def forward(self, src):
        """
        Forward pass through the TransformerVAE model.
        
        Args:
            src (tensor): Input trajectories (batch_size, sequence_length, 2).
            
        Returns:
            tuple: Encoded latent vector, mean, log variance, and decoded output.
        """
        enc_src, mu, logvar = self.encoder(src)
        out = self.decoder(enc_src)

        return enc_src, mu, logvar, out


class Bottleneck_Encoding(Leonardo):
    """
    Bottleneck encoding layer to compress the input into a latent vector.
    
    This module uses convolutional and dense layers to reduce the input
    to a smaller latent space representation.
    """
    def __init__(self, size, timesteps):
        """
        Initializes the bottleneck encoding module.
        
        Args:
            size (int): Dimensionality of input feature vectors.
            timesteps (int): Number of time steps in the input trajectories.
        """
        super().__init__(length_trajectory=200, latent_space=12)
        self.conv1 = nn.Conv1d(size, int(size / 4), kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(int(size / 4), int(size / 4), kernel_size=2, stride=2)
        self.linear1 = nn.Linear(int((size / 4) * (timesteps / 2)), 512)

    def forward(self, x):
        """
        Forward pass through the bottleneck encoder.
        
        Args:
            x (tensor): Input tensor (batch_size, embed_size, sequence_length).
            
        Returns:
            tensor: Encoded latent vector (batch_size, 512).
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.linear1(x))
        return x




class Bottleneck_Decoding(Leonardo):
    """
    Bottleneck decoding layer to upsample the latent vector back to input shape.
    
    This module uses dense and transpose convolutional layers to expand the
    latent vector back into a sequence suitable for the decoder blocks.
    """
    def __init__(self, size, timesteps):
        """
        Initializes the bottleneck decoding module.
        
        Args:
            size (int): Dimensionality of the input feature vectors (embedding size).
            timesteps (int): Number of time steps in the input trajectories.
        """
        super().__init__(length_trajectory=200, latent_space=12)
        self.linear1 = nn.Linear(1024, int((size / 4) * timesteps))
        self.convtranspose1 = nn.ConvTranspose1d(int(size / 4), size, kernel_size=7, padding=3)
        self.in_d = size
        self.timesteps = timesteps

    def forward(self, x):
        """
        Forward pass through the bottleneck decoder.
        
        Args:
            x (tensor): Latent vector (batch_size, latent_space_size).
            
        Returns:
            tensor: Upsampled output (batch_size, embed_size, sequence_length).
        """
        x = torch.relu(self.linear1(x))
        x = x.contiguous().view(-1, int(self.in_d / 4), int(self.timesteps))
        x = self.convtranspose1(x)
        return x
    
    

