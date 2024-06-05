# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:48:02 2024

@author: jduij
"""

import os
import torch
import tkinter as tk
from tkinter import filedialog
import shutil
import time
import sys

current _percent = dict()

def print_progress_bar(new_percent, prefix='', suffix='', decimals=0, length=40, fill='▓', id_key='progbar1'):
    """
    Call in a loop to create terminal progress bar.
    @params:
        new_percent - Required  : current value (float)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    global current_percent
    if new_percent == 0 or new_percent > current_percent.get(id_key, 0):
        current_percent[id_key] = new_percent
        percent_string = ("{0:." + str(decimals) + "f}").format(new_percent)
        filled_length = int(length * new_percent / 100)
        bar = fill * filled_length + '░' * (length - filled_length)
        sys.stdout.write(f'\r{prefix} {bar} {percent_string}% {suffix}')
        sys.stdout.flush()
        if new_percent >= 100.0:
            del current_percent[id_key]
            print()

def select_folder_and_check_files():
    # Set up the file dialog
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window

    # Prompt the user to select a folder
    folder_path = filedialog.askdirectory()

    # If the user cancels the dialog, return without doing anything
    if not folder_path:
        return

    # Create the corrupt_files_quarantine folder if it doesn't exist
    corrupt_folder = os.path.join(folder_path, "corrupt_files_quarantine")
    os.makedirs(corrupt_folder, exist_ok=True)

    # Scan for .pt files in the selected folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
    total_files = len(files)
    corrupt_files = 0

    start_time = time.time()

    for i, filename in enumerate(files):
        file_path = os.path.join(folder_path, filename)
        try:
            # Load the file using torch.load
            data = torch.load(file_path)

            # Check if the loaded data is a tensor with the correct shape and type
            if not isinstance(data, torch.Tensor):
                raise ValueError("Loaded data is not a tensor")
            if data.shape != (4, 6000) or data.dtype != torch.float32:
                raise ValueError("Tensor does not have the required shape (4, 6000) or type float32")

        except Exception as e:
            # If the file is corrupt or invalid, move it to the corrupt_files_quarantine folder
            shutil.move(file_path, corrupt_folder)
            corrupt_files += 1

        # Calculate progress and estimated remaining time
        progress_percent = ((i + 1) / total_files) * 100
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / (i + 1)) * total_files
        estimated_remaining_time = estimated_total_time - elapsed_time

        # Update the progress bar
        suffix = f"Corrupt: {corrupt_files}, Total: {total_files}, ETA: {estimated_remaining_time:.2f}s"
        print_progress_bar(round(progress_percent), prefix='Checking files:', suffix=suffix, id_key='file_check')

# Example usage
select_folder_and_check_files()
