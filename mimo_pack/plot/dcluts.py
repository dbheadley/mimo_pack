""" Plotting functions for dclut objects
Created on 2025-06-21
Drew Headley"""

import matplotlib.pyplot as plt
import dclut as dcl

def plot_probe(probe_dcl):
    """
    Plot the probe layout from a dclut object.
    
    Parameters
    ----------
    lfp_dcl : dclut object
        The dclut object containing the probe information.
    """
    # get the electrode positions
    x_pos = probe_dcl.scale_values('ch_x')
    y_pos = probe_dcl.scale_values('ch_y')
    ch_names = probe_dcl.scale_values('ch_order')

    # create a scatter plot of the electrode positions
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x_pos, y_pos, s=10, c='k', alpha=0.5)
    ax.set_aspect('equal')
    
    # plot the electrode numbers
    for i, (x, y) in enumerate(zip(x_pos, y_pos)):
        ax.text(x, y, ch_names[i], fontsize=8, ha='right', va='top')