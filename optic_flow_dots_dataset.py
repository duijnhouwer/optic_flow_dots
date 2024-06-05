# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 21:36:31 2024

@author: jduij
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import re


try:
    from line_profiler import LineProfiler
    profile = LineProfiler()
except ImportError:
    def profile(func):
        return func

# class SampleTransformer:
#     def __init__(self, params):
#         # Initialize any parameters needed for the transformation
#         self.params = params
#     def __call__(self, sample):
#         # Apply your transformation logic here
#         transformed_sample = sample.type(torch.float32)
#         return transformed_sample


class OpticFlowDotsDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path (str): Path to the folder containing files.
            transform (callable, optional): Optional transform to be applied to the features.
        """
        self.folder_path = folder_path
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith(".pt")]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    @profile
    def __getitem__(self, idx):
        if idx<0 or idx >= len(self.file_list):
            print(f"Index {idx} is out of range for dataset of length {len(self.data)}")
            raise IndexError("Index out of range")
        file_name = self.file_list[idx]
        file_path = os.path.join(self.folder_path, file_name)

        # Extract the translation and rotation parameters from the filename
        transrot_ruf = extract_target_response_from_filename(file_name)

        ## For now ignore the rotation, heading onlu
        #transrot_ruf = transrot_ruf[0:3]

        # Load the optic flow dots tensor from the file
        flow_tensor_4xN = torch.load(file_path)
        #flow_tensor_4xN.requires_grad = True

        # Select 2000 random dots (without replacement)
        num_columns = 2000
        random_indices = torch.randperm(flow_tensor_4xN.size(1))[:num_columns]
        flow_tensor_4xN = flow_tensor_4xN[:, random_indices]

        # Turn from 4xN matrix to 1x4*N matrix
        flow_tensor_1x4N = flow_tensor_4xN.view(1, -1)  # Flatten for fully connected layer

        return flow_tensor_1x4N, transrot_ruf

def extract_target_response_from_filename(input_string: str):
    # Use regular expression to find the part between brackets
    match = re.findall(r"[+-]?\d+(?=,|\])", input_string)

    if match:
        if len(match) == 6:
            # Create a PyTorch tensor from the floats
            floats = [float(num) for num in match]
            trans_rot_xyz = torch.tensor(floats, dtype=torch.float32)
            trans_rot_xyz /= 1e9 # convert from pico to milli units
            return trans_rot_xyz
        else:
            raise ValueError("The input string does not contain 6 floats.")
    else:
        raise ValueError("No valid pattern found in the input string ""{}"".".format(input_string))


# Example usage:
if __name__ == "__main__":
    data_folder_path = os.path.dirname(__file__)+'_data'
    dataset = OpticFlowDotsDataset(data_folder_path)

    # Create a DataLoader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Iterate over batches
    for batch_data, batch_features in dataloader:
        print("Batch data:", batch_data)
        print("Batch features:", batch_features)
