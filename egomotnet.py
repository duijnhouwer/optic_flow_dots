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
from datetime import datetime
from torchinfo import summary
import matplotlib.pyplot as plt
import random

# Initialize global figure and axis variables
figures={'loss_fig': None, 'loss_ax': None}

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
        if not type(init_dict['n_response_features']) == int or init_dict['n_response_features'] <= 0:
            raise ValueError("n_response_features must be a positive int")
        
        # Calculate the output shape of the Conv3d and MaxPool3d layers
        conv_bcfhw = lgg.conv3d_output_shape(init_dict['stimulus_shape'],init_dict['n_filters'],init_dict['filter_shape'])
        maxpool_fhw = (conv_bcfhw[2],1,1);
        pool_bcfhw = lgg.maxpool3d_output_shape(conv_bcfhw, maxpool_fhw, stride=(1,1,1))
        
        # Define the model architecture
        super(EgoMotNet, self).__init__()
        self.conv = nn.Conv3d(in_channels=init_dict['stimulus_shape'][1], out_channels=init_dict['n_filters'], kernel_size=init_dict['filter_shape'])
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=maxpool_fhw)
        self.fc = nn.Linear(in_features=math.prod(pool_bcfhw[1:]), out_features=init_dict['n_response_features'])

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.fc(x)
        
        return x


def train(dataset, checkpoint=None):
                           
    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Split dataset
    num_samples = len(dataset)
    train_size = int(0.70 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    if checkpoint == None:
        hyperparms = {'num_epochs': 100, 
                      'batch_size': 100, 
                      'learning_rate': 0.0005, 
                      'loss_fnc': nn.MSELoss()}
    else:
        hyperparms = checkpoint['hyperparms']

    # Initialize the data loaders
    train_loader = DataLoader(train_dataset, batch_size=hyperparms['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparms['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=1)

    if checkpoint == None:
        # Get one batch from the trainloader, so we know the sizes of the input to the model and the target
        for one_batch_data, one_batch_targets in train_loader:
            break
        stim_shape = tuple(one_batch_data.shape)
        n_resp = len(one_batch_targets[0])  
        init_dict = {'stimulus_shape': stim_shape,
                    'n_filters': 64,
                    'filter_shape': (stim_shape[2],11,11),
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
        log = {'time': [], 'epoch': [], 'val_loss': []}
        start_epoch = 0
    else:
        log = checkpoint['log']
        start_epoch = log['epoch'][-1]+1
    

    for epoch in range(start_epoch, hyperparms['num_epochs']):
        model.train()
        for batch_data, batch_targets in train_loader:
            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)  # Move data to the device         
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(batch_data.to(device)), batch_targets.to(device)).item() for batch_data, batch_targets in val_loader)
            log['time'].append(datetime.now())
            log['epoch'].append(epoch)
            log['val_loss'].append(val_loss)
            save_checkpoint(model, optimizer, hyperparms, init_dict, log)
            plot_loss(log)
            
        print(f"Epoch [{epoch}/{hyperparms['num_epochs']}], Train mLoss: {loss.item()*1000:.4f}, Val mLoss: {val_loss*1000 / len(val_loader):.4f}")

    print("Training complete!")

    # Evaluate on test data
    test_loss = 0.0
    with torch.no_grad():
        for batch_data, batch_targets in test_loader:
            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)  # Move data to the device
            outputs = model(batch_data)
            test_loss += criterion(outputs, batch_targets).item()

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
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return (model, optimizer)


def plot_loss(log, initialize=False):
    global figures
    if initialize or figures['loss_fig'] is None or figures['loss_ax'] is None:
        plt.close()  # Close the existing plot if any
        figures['loss_fig'], figures['loss_ax'] = plt.subplots(figsize=(10, 5))  # Create new figure and axes
         
    # Plot the new data
    plt.cla()
    figures['loss_ax'].set_xlabel('Epoch')  # Set x-axis label
    figures['loss_ax'].set_ylabel('Loss')  # Set y-axis label
    figures['loss_ax'].set_title('Loss vs. Epochs')  # Set title
    figures['loss_ax'].grid(True)  # Enable grid
    figures['loss_ax'].semilogy(log['epoch'], log['val_loss'], marker='o', linestyle='-', color='purple')
    plt.draw()  # Redraw the current figure
    plt.pause(0.1)  # Pause to update the plot

 
def main():
    torch.cuda.empty_cache()
    
    # Prevent Windows from going to sleep
    lgg.computer_sleep('prevent')
    
    data_folder_path = os.path.dirname(__file__)+'_data'
    dataset = OpticFlowDotsDataset(data_folder_path)
    
    # select a subset for testing
    dataset = Subset(dataset, random.sample(range(len(dataset) + 1), 500))
    
    checkpoint = load_checkpoint('MOST_RECENT_IN_DEFAULT_FOLDER')
    if checkpoint == None:
        print("Starting training from scratch ...")
        train(dataset)
    else:
        print("Continuing training at epoch {} ...".format(checkpoint['log']['epoch'][-1]))
        train(dataset,checkpoint)
    
    # Allow the system to sleep again
    lgg.computer_sleep('allow')
    
    torch.cuda.empty_cache()
    
    
if __name__=="__main__":
    main()
   
