# Unit data plotting functions
# Author: Drew Headley
# Date: 2025-06-15

import matplotlib.pyplot as plt
import numpy as np

def probe_units(spks_nap, jitter=10, labels=True, cmap='viridis', 
                spread_scale=1, ax=None, **kwargs):
    """
    Plots the distribution of spikes on a probe.

    Parameters
    ----------
    spks_nap : TsGroup object
        The spike data containing the units and their metadata 
        calculated by the pynapple_spikes_qc function.
    jitter : numeric, optional
        Amount of jitter to apply to the x and y positions of the
          units (default is 10).
    labels : bool, optional
        Whether to add labels to the axes and colorbar (default is True).
    cmap : str, optional
        Colormap to use for the scatter plot of SU rates (default is 'viridis').
    spread_scale : numeric, optional
        Scale factor for the size of the SU units in the scatter plot (default is 1).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (default is None, creates a new figure).
    **kwargs : additional keyword arguments
        Additional parameters for `plt.scatter`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes on which the waveforms were plotted.
    """
    
    # create a new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots()

    # check that metadata contains the required columns
    required_columns = ['x', 'y', 'class', 'rate', 'WaveformSpread']
    if not all(col in spks_nap.metadata.columns for col in required_columns):
        raise ValueError(f"Metadata must contain the following columns: {required_columns}")
    
    metadata = spks_nap.metadata
    mu_meta = metadata[metadata['class'] == 'MU']
    su_meta = metadata[metadata['class'] == 'SU']
    num_mu = mu_meta.shape[0]
    num_su = su_meta.shape[0]
    ax.scatter(mu_meta['x']+np.random.randn(num_mu)*jitter, 
               mu_meta['y']+np.random.randn(num_mu)*jitter,
               c='gray', s=20, marker='+', 
               label='MU units', **kwargs)
    ax.scatter(su_meta['x']+np.random.randn(num_su)*jitter, 
               su_meta['y']+np.random.randn(num_su)*jitter, 
               c=np.log10(su_meta['rate']), 
               s=su_meta['WaveformSpread']*spread_scale, 
               cmap=cmap, label='SU units', **kwargs)
    
    if np.max(mu_meta['x'])- np.min(mu_meta['x']) < 250:
        ax.set_xlim(np.min(mu_meta['x'])-125, np.max(mu_meta['x'])+125)
        
    ax.legend()
    cbar = plt.colorbar(ax.collections[1], ax=ax)
    ax.set_aspect('equal')
    
    if labels:
        cbar.set_label('log10(rate)')
        ax.set_title('Distribution of units on probe')
        ax.set_xlabel('x position (um)')
        ax.set_ylabel('y position (um)')
        
    return ax

