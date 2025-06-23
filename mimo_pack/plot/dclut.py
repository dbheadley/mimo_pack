""" Plotting functions for dclut objects
Created on 2025-06-21
Drew Headley"""

import matplotlib.pyplot as plt
import dclut as dcl
import numpy as np

def probe_layout(probe_dcl, chan_names=True, ax=None, scat_args={}, text_args={}):
    """
    Plot the probe layout from a dclut object.
    
    Parameters
    ----------
    lfp_dcl : dclut object
        The dclut object for a probe.
    chan_names : bool, optional
        Whether to display channel names as text labels (default is True).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (default is None, creates a new figure).
    scat_args : dict, optional
        Additional keyword arguments for the scatter plot (default is empty dict).
    text_args : dict, optional
        Additional keyword arguments for the text labels (default is empty dict).
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the probe layout plotted.
    """

    if ax is None:
        fig, ax = plt.subplots()

    # override default scatter and text arguments
    scat_args.setdefault('s', 10)
    scat_args.setdefault('c', 'k')
    scat_args.setdefault('alpha', 0.5)
    text_args.setdefault('fontsize', 5)
    text_args.setdefault('ha', 'right')
    text_args.setdefault('va', 'top')

    # get the electrode positions
    x_pos = probe_dcl.scale_values('ch_x')
    y_pos = probe_dcl.scale_values('ch_y')
    ch_idx = probe_dcl.scale_values('channel')

    # create a scatter plot of the electrode positions
    ax.scatter(x_pos, y_pos, **scat_args)
    ax.set_aspect('equal')
    
    # plot the electrode numbers
    if chan_names:
        for i, (x, y) in enumerate(zip(x_pos, y_pos)):
            ax.text(x, y, str(ch_idx[i]), **text_args)
    
    ax.set_xlabel('X Position (um)')
    ax.set_ylabel('Y Position (um)')

    return ax


def unit_lfp_mapping(spks, lfp, jitter=30, ax=None):
    """
    Plot mapping from single units to LFP channels with optional jitter for visibility.

    Parameters
    ----------
    spks : nap.TsGroup
        Pynapple TsGroup with .metadata DataFrame containing 'x', 'y', and 'lfp_channel'.
    lfp : xarray.DataArray
        LFP xarray object with channel dimension and .coords['ch_x'], .coords['ch_y'].
    jitter : float, optional
        Amount of jitter to add to unit positions for visibility (default: 30).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the mapping plotted.
    """

    if ax is None:
        fig, ax = plt.subplots()

    for unit in spks:
        unit_pos = spks.metadata.loc[unit, ['x', 'y']].values
        # Add random jitter for better visibility
        unit_pos = unit_pos + np.random.uniform(-jitter, jitter, size=2)
        lfp_ch = spks.metadata.loc[unit, 'lfp_channel']
        if np.isnan(lfp_ch):
            # plot the unit position only, with a green circle
            ax.plot(unit_pos[0], unit_pos[1], 'go', markersize=5, alpha=0.5)
            continue
        lfp_x = lfp.sel(channel=int(lfp_ch)).coords['ch_x'].values
        lfp_y = lfp.sel(channel=int(lfp_ch)).coords['ch_y'].values
        ax.annotate(
            '',
            xy=(lfp_x, lfp_y),
            xytext=(unit_pos[0], unit_pos[1]),
            arrowprops=dict(arrowstyle="->", color='k', lw=1, alpha=0.2)
        )
        ax.plot(unit_pos[0], unit_pos[1], 'bo', markersize=5, alpha=0.5)
        ax.plot(lfp_x, lfp_y, 'r+', markersize=5, alpha=0.5)
    ax.set_xlabel('X position (um)')
    ax.set_ylabel('Y position (um)')
    ax.set_title('Unit to LFP Channel Mapping')
    
    return ax