# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 22:14:38 2024

@author: jduij
"""

import re
import torch

def extract_trans_rot_xyz(input_string):
    # Use regular expression to find the part between brackets
    match = re.search(r"\[(-?\d+\.\d+(_-?\d+\.\d+)*)\]", input_string)
    
    if match:
        # Extract the matched string
        extracted_string = match.group(1)
        # Split the string by underscores and convert to floats
        floats = [float(x) for x in extracted_string.split('_')]
        
        if len(floats) == 6:
            # Create a PyTorch tensor from the floats
            trans_rot_xyz = torch.tensor(floats)
            return trans_rot_xyz
        else:
            raise ValueError("The input string does not contain 6 floats.")
    else:
        raise ValueError("No valid pattern found in the input string.")

# Example usage
input_string = "trxyz=[-10.123_99.12312231_8921.900_1231.2_182.44_12.0].pt"
try:
    result = extract_trans_rot_xyz(input_string)
    print(f"Extracted trans_rot_xyz: {result}")
except ValueError as e:
    print(f"Error: {e}")
