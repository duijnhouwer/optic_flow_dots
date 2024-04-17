# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:05:07 2024

@author: jduij
"""
import tkinter as tk
from tkinter import filedialog
from optic_flow_dots_dataset import extract_target_response_from_filename
import torch
import os

    
def select_file(initialdir: str=""):
    """Open a file dialog to select the tensor file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Ensure file dialog is on top
    if initialdir=="":
        file_path = filedialog.askopenfilename()
    else:
        file_path = filedialog.askopenfilename(initialdir=initialdir)
    root.destroy()  # Close the Tkinter root window after file selection
    return file_path
        
        
def load_stimulus_and_target_response(file_path: str=""):   
    if file_path=="":
        initialdir=os.path.dirname(__file__)+'_data'
        file_path=select_file(initialdir=initialdir)
        if not file_path:
            file_path="No file selected"        
    if file_path != "No file selected":
        stimulus = torch.load(file_path)
        target_response = extract_target_response_from_filename(input_string=file_path)
    else:
        stimulus, target_response = None
    return {'stimulus': stimulus, 'target_response': target_response, 'file_path': file_path}
    
    
def load_pytorch_model(file_path: str=""):
    if file_path=="":
        initialdir=os.path.dirname(__file__)+'_models'
        file_path=select_file(initialdir=initialdir)
        if not file_path:
            file_path="No file selected"        
    if file_path != "No file selected":
        train_stage = torch.load(file_path)
        return train_stage['model'].eval() # .eval() sets the model to evaluation mode
    else:
        model = None
    return model
    

def format_duration(seconds):
    seconds=round(seconds)
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    if days>0:
        str=f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds"
    elif hours>0:
        str=f"{hours} hours, {minutes} minutes, {seconds} seconds"
    elif minutes>0:
        str=f"{minutes} minutes, {seconds} seconds"
    else:
        str=f"{seconds} seconds"
    return str


def conv3d_output_shape(input_shape, n_output_chans, kernel_shape, padding=0, stride=1, dilation=1):
    """
    Calculate the output shape of a 3D convolution using tuples for dimensions and kernel sizes.

    Parameters:
    input_shape (tuple): Tuple containing the dimensions of the tensor in the order (nbatches, nchans, nframes, heipx, widpx):
        nbatches (int): Number of batches.
        nchans (int): Number of input channels.
        nframes (int): Number of frames (depth).
        heipx (int): Height of the input.
        widpx (int): Width of the input.
    kernel_shape (tuple): Tuple containing the kernel dimensions in the order (kernel_fr, kernel_widpx, kernel_heipx):
        kernel_fr (int): Size of the kernel along the frame (depth) dimension.
        kernel_widpx (int): Width of the kernel.
        kernel_heipx (int): Height of the kernel.
    padding (int or tuple): Padding added to all three sides of the input.
    stride (int or tuple): Stride of the convolution.
    dilation (int or tuple): Spacing between kernel elements.

    Returns:
    tuple: Shape of the output tensor (batch_size, num_output_channels, depth, height, width).
    """
    nbatches, nchans, nframes, heipx, widpx = input_shape
    kernel_fr, kernel_widpx, kernel_heipx = kernel_shape

    # Ensure padding, stride, and dilation are tuples of length 3
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    # Calculating the output dimensions for each spatial dimension
    def conv_output_dim(size, kernel_shape, padding, stride, dilation):
        return ((size + (2 * padding) - (dilation * (kernel_shape - 1)) - 1) // stride) + 1

    out_frames = conv_output_dim(nframes, kernel_fr, padding[0], stride[0], dilation[0])
    out_height = conv_output_dim(heipx, kernel_heipx, padding[1], stride[1], dilation[1])
    out_width = conv_output_dim(widpx, kernel_widpx, padding[2], stride[2], dilation[2])

    # Return the shape of the output tensor
    return (nbatches, n_output_chans, out_frames, out_height, out_width)


def maxpool3d_output_shape(input_size, kernel_size, stride=None, padding=0, dilation=1):
    """
    Calculate the output size of a tensor after applying nn.MaxPool3d.

    Parameters:
    input_size (tuple): Tuple of (N, C, D, H, W) representing the size of the input tensor.
    kernel_size (tuple): Tuple of (kD, kH, kW) representing the depth, height, and width of the kernel.
    stride (tuple): Tuple of (sD, sH, sW) representing the stride of the pool.
    padding (tuple or int): Padding on each dimension of the tensor.
    dilation (tuple or int): The spacing between kernel elements.

    Returns:
    tuple: The size of the tensor after max pooling.
    """
    # Default stride to kernel_size if not provided
    if stride is None:
        stride = kernel_size

    # Ensure stride, padding, and dilation are tuples
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    def pooled_output_size(dim_size, k, p, s, d):
        return ((dim_size + (2 * p) - d * (k - 1) - 1) // s) + 1

    N, C, D, H, W = input_size
    kD, kH, kW = kernel_size
    sD, sH, sW = stride
    pD, pH, pW = padding
    dD, dH, dW = dilation

    OD = pooled_output_size(D, kD, pD, sD, dD)
    OH = pooled_output_size(H, kH, pH, sH, dH)
    OW = pooled_output_size(W, kW, pW, sW, dW)

    return (N, C, OD, OH, OW)