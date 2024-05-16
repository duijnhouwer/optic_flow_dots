# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 21:36:31 2024

@author: jduij
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import re


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

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.folder_path, file_name)

        # Load the optic flow dots tensor from the file
        flow_tensor = torch.load(file_path)
        
        # Convert from uint8 [0..255] to float32 [0..1]
        flow_tensor = flow_tensor.to(torch.float32) / 255.0

        # Extract the translation and rotation parameters from the filename
        transrot_xyz = extract_target_response_from_filename(file_name)

        # Apply any specified transform
        if self.transform:
            transrot_xyz = self.transform(transrot_xyz)

        return flow_tensor, transrot_xyz
    
def extract_target_response_from_filename(input_string: str):
    # Use regular expression to find the part between brackets
    match = re.search(r"\[([-+]\d+\.\d+(_[-+]?\d+\.\d+)*)\]", input_string)

    if match:
        # Extract the matched string
        extracted_string = match.group(1)
        # Split the string by underscores and convert to floats
        floats = [float(x) for x in extracted_string.split('_')]

        if len(floats) == 6:
            # Create a PyTorch tensor from the floats
            trans_rot_xyz = torch.tensor(floats, dtype=torch.float32)
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
