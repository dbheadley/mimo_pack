""" 
Function for working with probe data
Created on 2025-06-21
Drew Headley
"""

import dclut as dcl
import numpy as np
import pandas as pd

def nearest_grid(probe_dcl, dx=250, dy=100):
    """
    Find the channels on a probe nearest to grid points.

    Parameters
    ----------
    probe_dcl : dclut.Probe
        The probe data containing x and y positions.
    dx : int, optional
        The step size for the x-axis (default is 250).
    dy : int, optional
        The step size for the y-axis (default is 100).

    Returns
    -------
    ch_near : np.ndarray
        The x and y positions of the channels nearest to the grid points.
    x_near : np.ndarray
        The x positions of the channels nearest to the grid points.
    y_near : np.ndarray
        The y positions of the channels nearest to the grid points.
    """
    
    x_pos = probe_dcl.scale_values('ch_x')
    y_pos = probe_dcl.scale_values('ch_y')
    ch_idx = probe_dcl.scale_values('channel')
    
    x_min, x_max = np.nanmin(x_pos), np.nanmax(x_pos)
    y_min, y_max = np.nanmin(y_pos), np.nanmax(y_pos)
    
    x_grid = np.arange(x_min, x_max, dx)
    y_grid = np.arange(y_min, y_max, dy)
    
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # create arrays to hold the nearest channel indices and positions
    ch_near = np.empty(X.size, dtype=ch_idx.dtype)
    x_near = np.empty(X.size, dtype=x_pos.dtype)
    y_near = np.empty(X.size, dtype=y_pos.dtype)

    for i, (x, y) in enumerate(zip(X.flatten(), Y.flatten())):
        # calculate the distance from each channel to the grid point
        distances = np.sqrt((x_pos - x)**2 + (y_pos - y)**2)
        nearest_idx = np.nanargmin(distances)
        
        # set the channel position to the grid point
        ch_near[i] = ch_idx[nearest_idx]
        x_near[i] = x_pos[nearest_idx]
        y_near[i] = y_pos[nearest_idx]

    return ch_near, x_near, y_near


def find_lfp_channel(spks, lfp, min_dist=100, max_dist=200):
    """
    Find the closest LFP channel for each unit on a probe.

    Parameters
    ----------
    spks : nap.TsGroup
        Pynapple TsGroup containing spike times for each unit
    lfp : xarray.DataArray
        LFP data with channel coordinates
    min_dist : int, optional
        Minimum distance to consider a channel as a candidate, by default 100
    max_dist : int, optional
        Maximum distance to consider a channel as a candidate, by default 200

    Returns
    -------
    nap.TsGroup
        Pynapple TsGroup with added LFP channel for each unit
    """

    # get the coordinates of the LFP channels
    lfp_coords = np.array([lfp.coords['ch_x'].values, 
                           lfp.coords['ch_y'].values]).T
    lfp_chans = lfp.coords['channel'].values
    # get the coordinates of the units
    unit_coords = spks.metadata[['x', 'y']].values

    unit_lfp_ch = pd.Series(np.nan, index=spks.metadata.index, name='lfp_channel')
    # calculate the distance between each unit and each LFP channel
    for i, unit_coord in enumerate(unit_coords):
        dist = np.linalg.norm(lfp_coords - unit_coord, axis=1)
        # find the closest LFP channel within the specified distance range
        candidates = np.where((dist >= min_dist) & (dist <= max_dist))[0]
        if len(candidates) > 0:
            closest_ch = candidates[np.argmin(dist[candidates])]
            unit_lfp_ch.iloc[i] = lfp_chans[closest_ch]
    
    # add the LFP channel to the metadata of the spike times
    spks.set_info({'lfp_channel': unit_lfp_ch})

    return spks