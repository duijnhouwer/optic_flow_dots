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
from datetime import datetime
import egomotnet_plot
import time

class EgoMotNet(nn.Module):

    def __init__(self, hparams: dict=None):

        # Check that the hyperparameter dictionary is valid
        verify_hparams(hparams)

        # Calculate the output shape of the Conv3d and MaxPool3d layers
        conv_out_bcfhw = lgg.conv3d_output_shape(hparams['X_shape'], hparams['n_filters']*hparams['n_frame_orientations'], hparams['filter_shape'])
        pool_kernel_fhw = hparams['pool_widhei'] #(conv_out_bcfhw[2], hparams['pool_widhei'], hparams['pool_widhei']) #hparams['pool_widhei'] #
        #pool_out_bcfhw = lgg.maxpool3d_output_shape(conv_out_bcfhw, pool_kernel_fhw)

        # Add hparam n_frame_orientations so it can be referenced in forward()
        self.n_frame_orientations = hparams['n_frame_orientations']

        # Define the model architecture
        super(EgoMotNet, self).__init__()
        self.conv = nn.Conv3d(hparams['X_shape'][1], hparams['n_filters'], hparams['filter_shape'])
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(pool_kernel_fhw)
        #self.fc1 = nn.Linear(math.prod(pool_out_bcfhw[1:]), hparams['fc1_n_out'])
        self.fc1 = nn.LazyLinear(hparams['fc1_n_out'])
        self.fc2 = nn.Linear(hparams['fc1_n_out'], hparams['fc2_n_out'])
        self.fc3 = nn.Linear(hparams['fc2_n_out'], hparams['final_n_out'])

    def forward(self, x):
        if self.n_frame_orientations == 1:
            x = self.conv(x)
        elif self.n_frame_orientations == 6:
            x0 = self.conv(x)
            x90 = batch_rotate(self.conv(batch_rotate(x, 90)), -90)
            x180 = batch_rotate(self.conv(batch_rotate(x, 180)), -180)
            x270 = batch_rotate(self.conv(batch_rotate(x, 270)), -270)
            xUD = batch_flipud(self.conv(batch_flipud(x)))
            xLR = batch_fliplr(self.conv(batch_fliplr(x)))
            x = torch.cat((x0, x90, x180, x270, xUD, xLR), dim=1)
        else:
            raise ValueError("n_flips_and_turns must be 1 or 6")
        x = x.squeeze(2)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)  # Flatten for fully connected layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def verify_hparams(hp):
    if hp is None:
        raise ValueError("Must provide initialization dictionary")
    if len(hp['X_shape']) != 5 or not all(isinstance(x, int) and x > 0 for x in hp['X_shape']):
        raise ValueError("X_shape must be a 5-element tuple of positive ints")
    if not isinstance(hp['n_filters'], int) or hp['n_filters'] <= 0:
        raise ValueError("n_filters must be a positive int")
    if len(hp['filter_shape']) != 3 or not all(isinstance(x, int) and x > 0 for x in hp['filter_shape']):
        raise ValueError("filter_shape must be a 3-element tuple of positive ints")
    if not isinstance(hp['pool_widhei'], int) or hp['pool_widhei'] <= 0:
        raise ValueError("pool_widhei must be a positive int")
    if not isinstance(hp['fc1_n_out'], int) or hp['fc1_n_out'] <= 0:
        raise ValueError("fc1_n_out must be a positive int")
    if not isinstance(hp['fc2_n_out'], int) or hp['fc2_n_out'] <= 0:
        raise ValueError("fc2_n_out must be a positive int")
    if not isinstance(hp['final_n_out'], int) or hp['final_n_out'] <= 0:
        raise ValueError("final_n_out must be a positive int")
    if hp['n_frame_orientations']!=1 and hp['n_frame_orientations']!=6:
        raise ValueError("n_frame_orientations must be 1 or 6")


def batch_rotate(movie_batch, angle_deg):
    # rotate the frames of the movies by angle_deg
    if angle_deg % 90 != 0:
        raise ValueError('angle_deg must be a multiple of 90')
    return torch.rot90(movie_batch, k=int(angle_deg / 90.0), dims=(3, 4))

def batch_flipud(movie_batch):
    return torch.flip(movie_batch, dims=[3])

def batch_fliplr(movie_batch):
    return torch.flip(movie_batch, dims=[4])

def create_data_loaders(data, batch_size):
    samples_n = len(data)
    train_n = int(0.70 * samples_n)
    val_n = int(0.15 * samples_n)
    test_n = samples_n - train_n - val_n
    train_data, val_data, test_data = random_split(data, (train_n, val_n, test_n))

    # Initialize the data loaders.
    train_loader = DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_data, batch_size, drop_last=True)
    test_loader = DataLoader(test_data, batch_size, drop_last=True)

    if len(train_loader) == 0:
        raise ValueError(f'Number of samples in train_data ({len(train_data)}) is less than the batch size ({train_loader.batch_size}).')
    if len(val_loader) == 0:
        raise ValueError(f'Number of samples in val_data ({len(val_data)}) is less than the batch size ({val_loader.batch_size}).')
    if len(test_loader) == 0:
        raise ValueError(f'Number of samples in test_data ({len(test_data)}) is less than the batch size ({test_loader.batch_size}).')

    return train_loader, val_loader, test_loader

def train(data, n_epochs=100, ckpt=None):
    model, optimizer, hparams, log, start_epoch = init_from_checkpoint(ckpt, data[0])
    train_loader, val_loader, test_loader = create_data_loaders(data, hparams['batch_size'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        for epoch in range(start_epoch, start_epoch + n_epochs):
            epoch_start_s = time.time()

            # Train the model
            model.train()  # Set the model to training mode
            total_train_loss = 0.0
            for batch_count, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)  # Move data to the device
                optimizer.zero_grad()
                train_loss = hparams['loss_fnc'](model(X), y)
                train_loss.backward()
                optimizer.step()
                total_train_loss += train_loss.item()
                lgg.print_progress_bar(batch_count + 1, len(train_loader), f'Epoch {epoch} training ...')
            mean_train_loss_per_movie = total_train_loss / (batch_count + 1)

            # Validate the model
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                total_val_loss = 0.0
                for batch_count, (X, y) in enumerate(val_loader):
                    X, y = X.to(device), y.to(device)  # Move data to the device
                    yHat = model(X)
                    total_val_loss += hparams['loss_fnc'](yHat, y).item()
                    lgg.print_progress_bar(batch_count + 1, len(val_loader), f'Epoch {epoch} validation .')
                log['time'].append(datetime.now())
                log['epoch'].append(epoch)
                log['val_loss'].append(total_val_loss / (batch_count + 1))
                log['train_loss'].append(mean_train_loss_per_movie)
                save_checkpoint(model, optimizer, hparams, log)
                egomotnet_plot.plot_progress(log)
            print(f"Duration of epoch {epoch}: {lgg.format_duration(time.time()-epoch_start_s)}.")
    except KeyboardInterrupt:
        print("\n*** Training canceled by user")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # Test the model
        n_rows = len(test_loader) * test_loader.batch_size
        test_result = {'loss': 0.0, 'y': torch.zeros(n_rows, 6), 'yHat': torch.zeros(n_rows, 6)}
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for batch_count, (X, y) in enumerate(test_loader):
                X, y = X.to(device), y.to(device)  # Move data to the device
                yHat = model(X)
                test_result['loss'] += hparams['loss_fnc'](yHat, y).item()
                test_result['y'][batch_count:batch_count + test_loader.batch_size] = y.cpu()
                test_result['yHat'][batch_count:batch_count + test_loader.batch_size] = yHat.cpu()
                lgg.print_progress_bar(batch_count + 1, len(test_loader), 'Testing the model ..')
        test_result['loss'] /= (batch_count + 1)
        filename = save_checkpoint(model, optimizer, hparams, log, test_result)
        egomotnet_plot.plot_test_result(filename)
        print(f"Mean test loss: {test_result['loss'] / len(test_loader):.9f}")

def save_checkpoint(model, optimizer, hparams, log, test_result=None):
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder = os.path.dirname(__file__) + '_models'
    mode = 'checkpoint' if test_result is None else 'finaltest'
    filename = os.path.join(folder, "{}_{}_epoch_{}.pth".format(mode, now_str, log['epoch'][-1]))
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'hparams': hparams,
        'log': log,
        'test_result': test_result
    }, filename)
    print("Saved: {}".format(filename))
    return filename

def load_checkpoint(file_path=""):
    modeldir = os.path.dirname(__file__) + '_models'
    if file_path == "":
        file_path = lgg.select_file(initialdir=modeldir)
        if not file_path:
            file_path = "No file selected"
    elif file_path == "MOST_RECENT_IN_DEFAULT_FOLDER":
        # Create a list of files matching 'checkpoint*.pth' and their modification times
        base_filenames = [(f, os.path.getmtime(os.path.join(modeldir, f)))
                          for f in os.listdir(modeldir)
                          if os.path.isfile(os.path.join(modeldir, f))
                          and fnmatch.fnmatch(f, 'checkpoint*.pth')]
        if base_filenames:
            # Sort files by modification time in descending order
            base_filenames.sort(key=lambda x: x[1], reverse=True)
            # Select the most recent file
            file_path = os.path.join(modeldir, base_filenames[0][0])
        else:
            print(f"No 'checkpoint*.pth' file found in '{modeldir}'")
            file_path = "No file selected"

    if file_path != "No file selected":
        try:
            ckpt = torch.load(file_path)
        except Exception as e:
            print(f"An error occurred while loading checkpoint {file_path}:\n\t{e}")
            ckpt = None
    else:
        ckpt = None
    return ckpt

def init_from_checkpoint(ckpt, datum=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ckpt is None:
        print("Creating a fresh random start point ...")
        ckpt = create_startpoint(datum)
    else:
        print("Loading epoch {} ckpt ...".format(ckpt['log']['epoch'][-1]))

    model = EgoMotNet(ckpt['hparams'])
    model.to(device)  # Move model device before you initialize the optimizer
    if ckpt['model_state'] is not None:
        model.load_state_dict(ckpt['model_state'])

    optimizer = optim.Adam(model.parameters(), lr=ckpt['hparams']['learning_rate'])
    if ckpt['optimizer_state'] is not None:
        optimizer.load_state_dict(ckpt['optimizer_state'])

    start_epoch = 0 if len(ckpt['log']['epoch']) == 0 else ckpt['log']['epoch'][-1] + 1

    return model, optimizer, ckpt['hparams'], ckpt['log'], start_epoch

def create_startpoint(datum):
    return {
        'hparams': {
            'batch_size': 12,
            'learning_rate': 0.001,
            'loss_fnc': nn.MSELoss(reduction='mean'),
            'X_shape': (1,) + tuple(datum[0].shape),
            'n_frame_orientations': 1,
            'n_filters': 256,
            'filter_shape': (datum[0].shape[1], 9, 9),
            'pool_widhei': 6,
            'fc1_n_out': 512,
            'fc2_n_out': 32,
            'final_n_out': len(datum[1])
        },
        'log': {'time': [], 'epoch': [], 'val_loss': [], 'train_loss': []},
        'model_state': None,
        'optimizer_state': None
    }

def main():
    try:
        torch.cuda.empty_cache()
        lgg.computer_sleep('disable')

        data_folder_path = os.path.dirname(__file__) + '_data'
        data = OpticFlowDotsDataset(data_folder_path)

        n_to_test = 1000 * 1000  # <=0 means test all stimuli
        if n_to_test > 0:
            data = Subset(data, random.sample(range(len(data) + 1), n_to_test))

        print("Training start time: "+lgg.now_string('date_time'))
        train(data, n_epochs=100, ckpt=load_checkpoint('MOST_RECENT_IN_DEFAULT_FOLDER'))
    except KeyboardInterrupt:
        print("Interrupted with CTRL-C")
    finally:
        lgg.computer_sleep('enable')
        torch.cuda.empty_cache()
        print("Training end time: "+lgg.now_string('date_time'))

if __name__ == "__main__":
    main()
