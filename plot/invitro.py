# Plots specifically for in vitro experiments
# Author: Drew Headley
# Date: 2025-07-10

import matplotlib.pyplot as plt
import xarray as xr
from mimo_pack.plot.map import wave_map_xr
import numpy as np


def focal_stim_map(rec: xr.DataArray, ax: plt.Axes = None,
                   window: list = [-0.05, 0.1], **kwargs) -> plt.Axes:
    """Plot the focal stimulation map for a recording.

    Parameters
    ----------
    rec : xr.DataArray
        The recording data with 'stim_x' and 'stim_y' coordinates.
    ax : plt.Axes, optional
        The axes to plot on. If None, a new figure and axes are created.
    window : list
        The time window to plot, specified as [start, end] in seconds.
    **kwargs : dict
        Additional keyword arguments to pass to the wave_map_xr function.

    Returns
    -------
    ax : plt.Axes
        The axes with the focal stimulation map plotted.
    """
    
    if ax is None:
        fig, ax = plt.subplots()

    # set default values for kwargs
    kwargs.setdefault('x_scale', 0.003)  
    kwargs.setdefault('y_scale', 0.2)    
    kwargs.setdefault('alpha', 0.5)

    # Ensure the recording has the required coordinates
    rec_plot = rec.sel(channel=0, time=slice(window[0], window[1]))
    rec_plot = rec_plot - rec_plot.mean(dim='time')  # remove DC offset
    wave_map_xr(rec_plot, x_coord='stim_x', y_coord='stim_y',
                ax=ax, **kwargs)
    ax.set_title('Waveform map for channel 0')
    ax.set_xlabel('Stimulus X position')
    ax.set_ylabel('Stimulus Y position')
    
    return ax
