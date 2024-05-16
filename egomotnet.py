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
import fnmatch
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
            
        if not type(init_dict['pool_widhei']) == int or init_dict['pool_widhei'] <= 0:
            raise ValueError("pool_widhei must be a positive int")    
            
        if not type(init_dict['fc1_n_out']) == int or init_dict['fc1_n_out'] <= 0:
            raise ValueError("fc1_n_out must be a positive int")
            
        if not type(init_dict['fc2_n_out']) == int or init_dict['fc2_n_out'] <= 0:
            raise ValueError("fc2_n_out must be a positive int")
            
        if not type(init_dict['final_n_out']) == int or init_dict['final_n_out'] <= 0:
            raise ValueError("final_n_out must be a positive int")
        
        # Calculate the output shape of the Conv3d and MaxPool3d layers
        conv_out_bcfhw = lgg.conv3d_output_shape(init_dict['stimulus_shape'], init_dict['n_filters']*4, init_dict['filter_shape'])
        pool_kernel_fhw = (conv_out_bcfhw[2], init_dict['pool_widhei'], init_dict['pool_widhei']);
        pool_out_bcfhw = lgg.maxpool3d_output_shape(conv_out_bcfhw, pool_kernel_fhw)
        
        # Define the model architecture
        super(EgoMotNet, self).__init__()
        self.conv = nn.Conv3d(in_channels=init_dict['stimulus_shape'][1], out_channels=init_dict['n_filters'], kernel_size=init_dict['filter_shape'])
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=pool_kernel_fhw)
        self.fc1 = nn.Linear(in_features=math.prod(pool_out_bcfhw[1:]), out_features=init_dict['fc1_n_out'])
        self.fc2 = nn.Linear(in_features=init_dict['fc1_n_out'], out_features=init_dict['fc2_n_out'])
        self.fc3 = nn.Linear(in_features=init_dict['fc2_n_out'], out_features=init_dict['final_n_out'])
    
    def forward(self, x):
        x0 = self.conv(x)
        x90 = rotate_batch(self.conv(rotate_batch(x,90)),-90)
        x180 = rotate_batch(self.conv(rotate_batch(x,180)),-180)
        x270 = rotate_batch(self.conv(rotate_batch(x,270)),-270)
        x = torch.cat((x0, x90, x180, x270), dim=1)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)  # Flatten for fully connected layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def rotate_batch(movie_batch, angle_deg):
    # rotate the frames of the movies by angle_deg.
    if angle_deg%90 != 0: raise ValueError('angle_deg must be a multiple of 90')
    return torch.rot90(movie_batch, k=int(angle_deg/90.0), dims=(3, 4))

def create_data_loaders(data,batch_size):
    samples_n = len(data)
    train_n = int(0.70 * samples_n)
    val_n = int(0.15 * samples_n)
    test_n = samples_n - train_n - val_n
    train_data, val_data, test_data = random_split(data, (train_n, val_n, test_n))

    # Initialize the data loaders. 
    # drop_last=True drops the last len(train_data)%batch_size movies to keep the mean loss per movie calculations accurate   
    train_loader = DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_data, batch_size, drop_last=True)
    test_loader = DataLoader(test_data, batch_size, drop_last=True) 
    
    if len(train_loader)==0:
        raise ValueError(f'Number of samples in train_data ({len(train_data)}) is less than the batchsize ({train_loader.batch_size}).')
    if len(val_loader)==0:
        raise ValueError(f'Number of samples in val_data ({len(val_data)}) is less than the batchsize ({val_loader.batch_size}).')
    if len(test_loader)==0:
        raise ValueError(f'Number of samples in test_data ({len(test_data)}) is less than the batchsize ({test_loader.batch_size}).')
    
    return (train_loader, val_loader, test_loader)


def train(data, n_epochs=100, checkpoint=None):
      
    model, optimizer, hyperparms, init_dict, log, start_epoch = init_from_checkpoint(checkpoint, data[0])                  
    train_loader, val_loader, test_loader = create_data_loaders(data,hyperparms['batch_size'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    
    try:
        for epoch in range(start_epoch,start_epoch+n_epochs):
            # Train the model
            model.train() # Set the model to training mode
            total_train_loss = 0.0
            for batch_count, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)  # Move data to the device         
                optimizer.zero_grad()
                train_loss = hyperparms['loss_fnc'](model(X), y)
                train_loss.backward()
                optimizer.step()
                total_train_loss += train_loss.item()
                lgg.print_progress_bar(batch_count+1,len(train_loader),f'Epoch {epoch} training ...')
            mean_train_loss_per_movie = total_train_loss / (batch_count+1)
            
            # Validate the model
            model.eval() # Set the model to evaluation mode
            with torch.no_grad():
                total_val_loss = 0.0
                for batch_count, (X, y) in enumerate(val_loader):
                    X, y = X.to(device), y.to(device)  # Move data to the device 
                    yHat = model(X)
                    total_val_loss += hyperparms['loss_fnc'](yHat, y).item()
                    lgg.print_progress_bar(batch_count+1,len(val_loader),f'Epoch {epoch} validation .')
                log['time'].append(datetime.now())
                log['epoch'].append(epoch)  
                log['val_loss'].append(total_val_loss / (batch_count+1))
                log['train_loss'].append(mean_train_loss_per_movie)
                save_checkpoint(model, optimizer, hyperparms, init_dict, log)
                egomotnet_plot.plot_progress(log)      
    except KeyboardInterrupt:
        print("Training canceled by user")
    except Exception as e:
        print(f"Error during egomotnet.train(): {e}")
    finally:
        n_rows = len(test_loader)*test_loader.batch_size
        test_result = {'loss': 0.0, 'y': torch.zeros(n_rows, 6), 'yHat': torch.zeros(n_rows, 6)}
        model.eval() # Set the model to evaluation mode
        with torch.no_grad():
            for batch_count, (X, y) in enumerate(test_loader):
                X, y = X.to(device), y.to(device)  # Move data to the device
                yHat = model(X)
                test_result['loss'] += hyperparms['loss_fnc'](yHat, y).item()
                test_result['y'][batch_count:batch_count+test_loader.batch_size] = y.cpu()
                test_result['yHat'][batch_count:batch_count+test_loader.batch_size] = yHat.cpu()
                lgg.print_progress_bar(batch_count+1,len(test_loader),f'Testing the model ..')
        test_result['loss'] /= (batch_count+1)
        filename = save_checkpoint(model, optimizer, hyperparms, init_dict, log, test_result)
        egomotnet_plot.plot_test_result(filename)
        print(f"Mean test loss: {test_result['loss'] / len(test_loader):.9f}")


def save_checkpoint(model, optimizer, hyperparms, init_dict, log, test_result=None):
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder = os.path.dirname(__file__)+'_models'
    mode = 'checkpoint' if test_result==None else 'finaltest'    
    filename = os.path.join(folder,"{}_{}_epoch_{}.pth".format(mode,now_str,log['epoch'][-1]))
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'hyperparms': hyperparms,
        'init_dict': init_dict, 
        'log': log,
        'test_result': test_result
         }, filename)
    print("Saved: {}".format(filename))
    return filename
    

def load_checkpoint(file_path: str=""):
    default_folder = os.path.dirname(__file__)+'_models'
    if file_path == "":
        file_path = lgg.select_file(initialdir=default_folder)
        if not file_path:
            file_path = "No file selected"       
    elif file_path == "MOST_RECENT_IN_DEFAULT_FOLDER":
        # Create a list of files matching 'checkpoint*.pth' along with their modification times 
        base_filenames = [(f, os.path.getmtime(os.path.join(default_folder, f))) 
                          for f in os.listdir(default_folder) if os.path.isfile(os.path.join(default_folder, f)) and fnmatch.fnmatch(f,'checkpoint*.pth')]
        if base_filenames:
            # Sort files by modification time in descending order
            base_filenames.sort(key=lambda x: x[1], reverse=True) 
            # Select the most recent file
            file_path = os.path.join(default_folder,base_filenames[0][0])
        else:
            print(f"No 'checkpoint*.pth' file found in '{default_folder}'")
            file_path = "No file selected"    
            
    if file_path != "No file selected":
        try:
            checkpoint = torch.load(file_path)
        except Exception as e:
            print(f"An error occurred while loading checkpoint {file_path}:\n\t{e}")
            checkpoint = None
    else:
        checkpoint = None
    return checkpoint


def init_from_checkpoint(cp, datum):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if cp == None:
        print("Starting the training from scratch ...")
        cp = create_startpoint(datum)
    else:
        print("Resuming training at epoch {} ...".format(cp['log']['epoch'][-1]))

    model = EgoMotNet(cp['init_dict'])
    model.to(device) # Move model device before you initialize the optimizer https://github.com/pytorch/pytorch/issues/2830#issuecomment-701138347
    if cp['model_state'] != None:
        model.load_state_dict(cp['model_state'])

    
    optimizer = optim.Adam(model.parameters(), lr=cp['hyperparms']['learning_rate'])
    if cp['optimizer_state'] != None:
        optimizer.load_state_dict(cp['optimizer_state'])
    
    start_epoch = 0 if len(cp['log']['epoch'])==0 else cp['log']['epoch'][-1]+1
    
    return (model, optimizer, cp['hyperparms'], cp['init_dict'], cp['log'], start_epoch)


def create_startpoint(datum) -> dict:
    return {'hyperparms': {'batch_size': 30, 
                           'learning_rate': 0.001, 
                           'loss_fnc': nn.MSELoss(reduction='mean')}
            ,'init_dict': {'stimulus_shape': (1,)+tuple(datum[0].shape),
                           'n_filters': 32,
                           'filter_shape': (datum[0].shape[1],13,13),
                           'pool_widhei': 6,
                           'fc1_n_out': 384,
                           'fc2_n_out': 60,
                           'final_n_out': len(datum[1])}
            ,'log': {'time': [], 'epoch': [], 'val_loss': [], 'train_loss': []}
            ,'model_state': None
            ,'optimizer_state': None }
    
 
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
        n_to_test = 30*1000 # <=0 means test all stimuli
        if n_to_test>0:
            data = Subset(data, random.sample(range(len(data) + 1), n_to_test))
       
        # Resume training from latest checkpoint
        train(data,n_epochs=10,checkpoint=load_checkpoint('MOST_RECENT_IN_DEFAULT_FOLDER'))
    except KeyboardInterrupt:
        print("Interupted with CTRL-C")
    finally:
        lgg.computer_sleep('enable')
        # Allow the system to sleep again
        lgg.computer_sleep('enable')
        torch.cuda.empty_cache()
        
    
if __name__=="__main__":
    main()
   
