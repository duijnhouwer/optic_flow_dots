# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 13:08:52 2024

@author: jduij
"""

import tkinter as tk
from tkinter import filedialog
import os
import shutil

def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select Data Folder")
    return folder_selected

def select_file():
    root = tk.Tk()
    root.withdraw()
    file_selected = filedialog.askopenfilename(title="Select Text File", filetypes=[("Text Files", "*.txt")])
    return file_selected

def move_files_to_corrupt_folder(data_folder, text_file):
    # Create the corrupt_Xy_files folder if it doesn't exist
    corrupt_folder = os.path.join(os.getcwd(), "corrupt_Xy_files")
    os.makedirs(corrupt_folder, exist_ok=True)

    # Read filenames from the text file
    with open(text_file, 'r') as file:
        filenames = file.readlines()

    # Move each file listed in the text file to the corrupt_Xy_files folder
    for filename in filenames:
        filename = filename.strip()
        file_path = os.path.join(data_folder, filename)
        if os.path.isfile(file_path):
            shutil.move(file_path, corrupt_folder)
            print(f"Moved: {file_path}")
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    data_folder = select_folder()
    if not data_folder:
        print("No folder selected. Exiting...")
    else:
        text_file = select_file()
        if not text_file:
            print("No file selected. Exiting...")
        else:
            move_files_to_corrupt_folder(data_folder, text_file)
