# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:58:20 2024

@author: jduij
"""

import glob
import os
from send2trash import send2trash
import inspect
from pathlib import Path


def delete_checkpoints():
    # Define the path to the folder where the models are by default
    
    this_file_path = inspect.getfile(lambda: None)
    path_parts = list(Path(this_file_path).parts) 
    models_folder_path = os.path.join(*path_parts[:-1])+'_models' # e.g. 'c:\\users\\jduij\\documents\\github\\optic_flow_dots_models'
    
    
    # Construct the search pattern to find all .pth files
    search_pattern = os.path.join(models_folder_path, 'checkpoint*.pth')
    
    # Use glob to find all .pth files in the folder
    checkpoint_files = glob.glob(search_pattern)
    
    if len(checkpoint_files)==0:
        print(f"No 'checkpoint*.pth' files to delete in '{models_folder_path}'")
    else:    
        # Loop through the list of .pth files and send each one to the Recycle Bin
        for file_path in checkpoint_files:
            send2trash(file_path)
            print(f"Sent to Recycle Bin: {file_path}")
            
            
if __name__ == "__main__":
    delete_checkpoints()             