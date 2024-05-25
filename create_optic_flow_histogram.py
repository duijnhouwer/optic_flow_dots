# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:27:05 2024

@author: jduij
"""

import numpy as np
import matplotlib.pyplot as plt
import optic_flow_dots

def plot_histogram_with_statistics(data, xlabel='Value'):
    # Ensure data is reshaped correctly, it should be flat for histogram plotting
    data = data.flatten()

    # Calculate statistics
    mean_value = np.mean(data)
    median_value = np.median(data)
    min_value = np.min(data)
    max_value = np.max(data)
    percentile_5 = np.percentile(data, 5)
    percentile_95 = np.percentile(data, 95)
    percentile_99 = np.percentile(data, 99)  # Calculate 99th percentile

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, color='lightblue', alpha=0.7, label='Data Histogram')

    # Add vertical lines for each statistic
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
    plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')
    plt.axvline(min_value, color='gray', linestyle='dashed', linewidth=2, label=f'Min: {min_value:.2f}')
    plt.axvline(max_value, color='black', linestyle='dashed', linewidth=2, label=f'Max: {max_value:.2f}')
    plt.axvline(percentile_5, color='purple', linestyle='dashed', linewidth=2, label=f'5th Percentile: {percentile_5:.2f}')
    plt.axvline(percentile_95, color='orange', linestyle='dashed', linewidth=2, label=f'95th Percentile: {percentile_95:.2f}')
    plt.axvline(percentile_99, color='blue', linestyle='dashed', linewidth=2, label=f'99th Percentile: {percentile_99:.2f}')

    # Add legend
    plt.legend()

    # Add titles and labels
    plt.title('Histogram with Statistical Annotations')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()



def get_step_sizes(n_steps: int=1000):
    deltas_px = np.full((n_steps, 1), np.nan)
    trans_speed_minmax=np.array([0.02,0.02])
    rot_dpf_minmax=np.array([0,0])

    delta_count = 0
    while True:

        # Pick a rotation and translation speed
        trans_speed = np.min(trans_speed_minmax) + np.random.sample() * trans_speed_minmax.ptp()
        rot_dpf = np.min(rot_dpf_minmax) + np.random.sample() * rot_dpf_minmax.ptp()

        # Create the translation and rotation vectors
        trans_ruf = optic_flow_dots.random_unit_vector(3) * trans_speed # ruf: right up front
        rot_ruf = optic_flow_dots.random_unit_vector(3) * rot_dpf

        delta_px = optic_flow_dots.generate_movie(trans_ruf=trans_ruf,
                                       rot_ruf=rot_ruf,
                                       n_dots=1,
                                       n_frames=2,
                                       wid_px=300,
                                       hei_px=300)

        if delta_px is None:
            continue
        else:
            deltas_px[delta_count] = delta_px
            delta_count += 1
            print(f'Found {delta_count} of {n_steps} displacements')
        if delta_count >= n_steps:
            print('[-: Done :-]')
            return deltas_px

if __name__=="__main__":
    step_sizes = get_step_sizes(n_steps=10000)
    plot_histogram_with_statistics(step_sizes, xlabel='Step size (px)')
