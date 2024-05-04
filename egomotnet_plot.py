# -*- coding: utf-8 -*-
"""
Created on Thu May  2 22:02:39 2024

@author: jduij
"""

import torch
import matplotlib.pyplot as plt


# Initialize global figure and axis variables
figures={'loss_fig': None, 'loss_ax': None}


def delta_deg(yHat: torch.Tensor, y: torch.Tensor) -> dict:
    # Check if inputs are PyTorch tensors
    if not (isinstance(yHat, torch.Tensor) and isinstance(y, torch.Tensor)):
        raise TypeError("Both yHat and y must be PyTorch tensors.")

    # Returns the angles between the predicted and the target rotation and translation vectors
    #
    # Extract the translation vectors (first 3 columns)
    trans_y = y[:, 0:3]
    trans_yHat = yHat[:, 0:3]

    # Extract the rotation vectors (last 3 columns)
    rot_y = y[:, 3:6]
    rot_yHat = yHat[:, 3:6]

    # Helper function to calculate the angle in degrees between vectors
    def angle_in_degrees(v1, v2):
        dot_product = torch.sum(v1 * v2, dim=1)
        norms_v1 = torch.norm(v1, dim=1)
        norms_v2 = torch.norm(v2, dim=1)
        cos_theta = dot_product / (norms_v1 * norms_v2)
        # Clamp the cosine values to the valid range for arccos
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        theta = torch.rad2deg(torch.acos(cos_theta))
        return theta

    # Calculate the angles for translation and rotation vectors
    t = angle_in_degrees(trans_y, trans_yHat)
    r = angle_in_degrees(rot_y, rot_yHat)

    out = {'trans': t, 'rot': r}

    return out

#def plot_angular_error_distributions(log, initialize=False):
    



def plot_progress(log, initialize=False):
    global figures
    if initialize or figures['loss_fig'] is None or figures['loss_ax'] is None:
        plt.close()  # Close the existing plot if any
        figures['loss_fig'], figures['loss_ax'] = plt.subplots(figsize=(10, 5))  # Create new figure and axes
         
    # Plot the new data
    plt.cla()
    figures['loss_ax'].set_xlabel('Epoch')  # Set x-axis label
    figures['loss_ax'].set_ylabel('Mean loss per movie')  # Set y-axis label
    figures['loss_ax'].set_title('Loss vs. Epochs')  # Set title
    figures['loss_ax'].grid(True)  # Enable grid
    figures['loss_ax'].semilogy(log['epoch'], log['val_loss'], marker='o', linestyle='-', color='violet', label='validation')
    figures['loss_ax'].semilogy(log['epoch'], log['train_loss'], marker='o', linestyle='-', color='teal', label='training')
    plt.legend(loc='upper right') 
    plt.draw()  # Redraw the current figure
    plt.pause(0.1)  # Pause to update the plot
    