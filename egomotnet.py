# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 23:13:47 2024

@author: jduij
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from optic_flow_dots_dataset import OpticFlowDotsDataset
import the_luggage as lgg
import math
import os
import random
import inspect
from datetime import datetime
from torchinfo import summary
import egomotnet_plot


class EgoMotNet(nn.Module):
    def __init__(self, init_dict: dict=None):
    
        # Check the inputs
        if init_dict == None:
            raise ValueError("Must provide initialization dictionary")
            
        if len(init_dict['stimulus_shape']) != 5 or not all(isinstance(x, int) and x > 0 for x in init_dict['stimulus_shape']):
            raise ValueError("stimulus_shape must be a 5-element tuple of positive ints")  
            
        if not type(init_dict['n_filters']) == int or init_dict['n_filters'] <= 0:
            raise ValueError("nfilters must be a positive int")
            
        if len(init_dict['filter_shape']) != 3 or not all(isinstance(x, int) and x > 0 for x in init_dict['filter_shape']):
            raise ValueError("filter_shape must be a 3-element tuple of positive ints")  
            
        if not type(init_dict['max_pool_widhei']) == int or init_dict['max_pool_widhei'] <= 0:
            raise ValueError("max_pool_widhei must be a positive int")    
            
        if not type(init_dict['n_response_features']) == int or init_dict['n_response_features'] <= 0:
            raise ValueError("n_response_features must be a positive int")
        
        # Calculate the output shape of the Conv3d and MaxPool3d layers
        conv_out_bcfhw = lgg.conv3d_output_shape(init_dict['stimulus_shape'], init_dict['n_filters']*4, init_dict['filter_shape'])
        pool_kernel_fhw = (conv_out_bcfhw[2],2,2) #(conv_out_bcfhw[2], init_dict['max_pool_widhei'], init_dict['max_pool_widhei']);
        pool_out_bcfhw = lgg.maxpool3d_output_shape(conv_out_bcfhw, pool_kernel_fhw)
        
        # Define the model architecture
        super(EgoMotNet, self).__init__()
        self.conv = nn.Conv3d(in_channels=init_dict['stimulus_shape'][1], out_channels=init_dict['n_filters'], kernel_size=init_dict['filter_shape'])
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel_fhw)
        self.fc = nn.Linear(in_features=math.prod(pool_out_bcfhw[1:]), out_features=init_dict['n_response_features'])

    
    def forward(self, x):
        #rotmov = lambda x, angle_deg: torch.rot90(x, k=int(angle_deg/90), dims=(3, 4))
        x0 = self.conv(x)
        x90 = rotmov(self.conv(rotmov(x,90)),-90)
        x180 = rotmov(self.conv(rotmov(x,180)),-180)
        x270 = rotmov(self.conv(rotmov(x,270)),-270)
        x = torch.cat((x0, x90, x180, x270), dim=1)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)  # Flatten for fully connected layer
        x = self.fc(x)
        return x


def rotmov(movie_batch, angle_deg):
    # rotate the frames of the movies by angle_deg.
    if angle_deg%90 != 0: raise ValueError('angle_deg must be a multiple of 90')
    return torch.rot90(movie_batch, k=int(angle_deg/90), dims=(3, 4))


def train(data, checkpoint=None):
    
    # Load the hyperparms from the checkpoint, or define them here if no checkpoint was provided
    if checkpoint == None:
        hyperparms = {'num_epochs': 100, 
                      'batch_size': 10, 
                      'learning_rate': 0.001, 
                      'loss_fnc': nn.MSELoss(reduction='mean')}
    else:
        hyperparms = checkpoint['hyperparms']
                           
    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Split data
    samples_n = len(data)
    init_n = 1
    train_n = int(0.70 * samples_n - init_n)
    val_n = int(0.15 * samples_n - init_n)
    test_n = samples_n - init_n - train_n - val_n
    init_data, train_data, val_data, test_data = random_split(data, (init_n, train_n, val_n, test_n))

    # Initialize the data loaders. 
    # drop_last=True drops the last len(train_data)%batch_size movies to keep the mean loss per movie calculations accurate   
    train_loader = DataLoader(train_data, batch_size=hyperparms['batch_size'], drop_last=True, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=hyperparms['batch_size'], drop_last=True)
    test_loader = DataLoader(test_data, batch_size=1) # no drop_last needed if batch_size = 1
    init_loader = DataLoader(init_data, batch_size=1)
    
    if len(train_data)==0:
        raise ValueError('Number of samples in train_data ({}) is less than the batchsize ({}).'.format(len(train_data),train_loader.batch_size))
    if len(val_data)==0:
        raise ValueError('Number of samples in val_data ({}) is less than the batchsize ({}).'.format(len(val_data),val_loader.batch_size))
    
    if checkpoint == None:
        # Get one batch from the trainloader, so we know the sizes of the input to the model and the target
        X, y = next(iter(init_loader))
        stim_shape = tuple(X.shape)
        n_resp = len(y[0])  
        init_dict = {'stimulus_shape': stim_shape,
                    'n_filters': 32,
                    'filter_shape': (stim_shape[2],11,11),
                    'max_pool_widhei': 5,
                    'n_response_features': n_resp}
    else:
        init_dict = checkpoint['init_dict']
    
    # Initialize the neural network
    model = EgoMotNet(init_dict)
    if checkpoint != None:
        model.load_state_dict(checkpoint['model_state'])
    summary(model, input_size=init_dict['stimulus_shape'])    
    model.to(device)
        
    # Initialize the optimizer  
    optimizer = optim.Adam(model.parameters(), lr=hyperparms['learning_rate'])
    if checkpoint != None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    # Initialize the criterion  
    criterion = hyperparms['loss_fnc']
    
    # Initialize the starting epoch number and the log
    if checkpoint == None:
        log = {'time': [], 'epoch': [], 'val_loss': [], 'train_loss': [], 'val_trans_delta_deg': [], 'val_rot_delta_deg': []}
        start_epoch = 0
    else:
        log = checkpoint['log']
        start_epoch = log['epoch'][-1]+1
    

    for epoch in range(start_epoch, hyperparms['num_epochs']):
        model.train() # Set the model to training mode
        total_train_loss = 0.0
        batch_count  = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)  # Move data to the device         
            optimizer.zero_grad()
            train_loss = criterion(model(X), y)
            train_loss.backward()
            optimizer.step()
            total_train_loss += train_loss.item()
            batch_count += 1
        mean_train_loss_per_movie = total_train_loss / batch_count
           
        model.eval() # Set the model to evaluation mode
        with torch.no_grad():
            total_val_loss = 0.0
            batch_count = 0
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)  # Move data to the device 
                yHat = model(X)
                val_loss = criterion(yHat, y)
                val_delta_deg = egomotnet_plot.delta_deg(yHat, y)
                total_val_loss += val_loss.item()
                batch_count += 1
            mean_val_loss_per_movie = total_val_loss / batch_count
            log['time'].append(datetime.now())
            log['epoch'].append(epoch)
            log['val_loss'].append(mean_val_loss_per_movie)
            log['train_loss'].append(mean_train_loss_per_movie)
            log['val_trans_delta_deg'].append(val_delta_deg['trans'])
            log['val_rot_delta_deg'].append(val_delta_deg['rot'])
            save_checkpoint(model, optimizer, hyperparms, init_dict, log)
            egomotnet_plot.plot_progress(log)

    print("Training complete!")

    # Evaluate on test data
    test_loss = 0.0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)  # Move data to the device
            test_loss += criterion(model(X), y).item()

    print(f"Test mLoss: {test_loss*1000 / len(test_loader):.4f}")



def save_checkpoint(model, optimizer, hyperparms, init_dict, log):
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder = os.path.dirname(__file__)+'_models'
    checkpoint_filename = os.path.join(folder,"checkpoint_{}_epoch_{}.pth".format(now_str,log['epoch'][-1]))
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'hyperparms': hyperparms,
        'init_dict': init_dict, 
        'log': log
         }, checkpoint_filename)
    print("Saved: {}".format(checkpoint_filename))
    

def load_checkpoint(file_path: str=""):
    default_folder = os.path.dirname(__file__)+'_models'
    if file_path == "":
        file_path = lgg.select_file(initialdir=default_folder)
        if not file_path:
            file_path = "No file selected"       
    elif file_path == "MOST_RECENT_IN_DEFAULT_FOLDER":
        # Create a list of files ending with '.pth' along with their paths and modification times
        files = [(os.path.join(default_folder, f), os.path.getmtime(os.path.join(default_folder, f)))
                 for f in os.listdir(default_folder) if os.path.isfile(os.path.join(default_folder, f)) and f.endswith('.pth')]
        if files:
            # Sort files by modification time in descending order
            files.sort(key=lambda x: x[1], reverse=True) 
            # Select the most recent file
            file_path = files[0][0]
        else:
            print('No .pth file found in {}'.format(default_folder))
            file_path = "No file selected"    
            
    if file_path != "No file selected":
        checkpoint = torch.load(file_path)
    else:
        checkpoint = None
    return checkpoint


def init_from_checkpoint(checkpoint):
    model = EgoMotNet(checkpoint['init_dict'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval() # Set the model to evaluation mode
    optimizer = optim.Adam(model.parameters(), lr=checkpoint['hyperparms']['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return (model, optimizer)





 
def main():
    
    try: 
        # Try release some GPU RAM. But the best way is to restart Spyder
        torch.cuda.empty_cache()
        # Prevent Windows from going to sleep
        lgg.computer_sleep('disable')
    
        # Load the dataset from the default folder
        data_folder_path = os.path.dirname(__file__)+'_data'
        data = OpticFlowDotsDataset(data_folder_path)
    
        # select a subset of data for quick testing
        data = Subset(data, random.sample(range(len(data) + 1), 100))
    
        checkpoint = load_checkpoint('MOST_RECENT_IN_DEFAULT_FOLDER')
        if checkpoint == None:
            print("Starting the training from scratch ...")
            train(data)
            # note: default hyperparameters are defined in train()
        else:
            print("Resuming training at epoch {} ...".format(checkpoint['log']['epoch'][-1]))
            train(data,checkpoint)
    except KeyboardInterrupt:
        print("Interupted with CTRL-C")
    finally:
        lgg.computer_sleep('enable')
        # Allow the system to sleep again
        lgg.computer_sleep('enable')
        torch.cuda.empty_cache()
        
    
if __name__=="__main__":
    main()
   
