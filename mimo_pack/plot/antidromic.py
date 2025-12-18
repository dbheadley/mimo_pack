# Antidromic plots

from mimo_pack.plot.xarray import composite_xr, pcolormesh_xr
import matplotlib.pyplot as plt

def plot_antidromic_summary(summary_data, current_uA, region, fig=None):
    """Plots the antidromic response summary including AP envelope, LFP, and CSD.
    This function generates a figure with three main sections, one for each data
    type: Action Potential (AP) envelope, Local Field Potential (LFP), and
    Current Source Density (CSD). Each section is further divided into subplots
    based on the shank of the recording probe, as defined in the input DataArrays.
    Parameters
    ----------
    summary_data : dict
        A dictionary containing the summary data. It must have the keys 'ap',
        'lfp', and 'csd', where each value is an xarray DataArray containing
        the corresponding data.
    current_uA : int or float
        The stimulation current in microamperes (uA) to be displayed in the
        figure's main title.
    region : str
        The brain region stimulated (e.g., 'AC', 'M1'), which will be included
        in the figure's main title.
    fig : matplotlib.figure.Figure, optional
        An existing matplotlib Figure object to plot on. If None, a new
        figure is created. Defaults to None.
    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the generated summary plots.
    Notes
    -----
    - The function relies on the custom plotting functions `composite_xr` and
      `pcolormesh_xr`, which are expected to handle xarray DataArrays.
    - The input DataArrays in `summary_data` are expected to have coordinates
      like 'ch_shank', 'ch_x', 'ch_y', and 'time' for proper plotting.
    - The layout is managed using `fig.subfigures` and `layout='constrained'`
      to organize the plots for AP, LFP, and CSD vertically and prevent titles
      from overlapping.
    """
 
    ap_env = summary_data['ap']
    lfp_mean = summary_data['lfp']
    csd_mean = summary_data['csd']

    # 1. Initialize with constrained layout to prevent overlapping titles
    if fig is None:
        fig = plt.figure(figsize=(8, 10), layout='constrained')
    
    fig.suptitle(f'Antidromic Response - Stim: {current_uA} uA {region}', fontsize=16)
    
    subfigs = fig.subfigures(nrows=3, ncols=1)

    # --- AP Envelope plot ---
    subfigs[0].suptitle('AP Envelope', weight='bold')
    # 2. Add sharey=True to link axes and hide right-side ticks
    axs = subfigs[0].subplots(1, 2)
    composite_xr(ap_env, subplot_coord='ch_shank', color_coord='ch_x', 
                 background=[0, 0, 0], axs=axs)
    
    for i, ax in enumerate(axs.flatten()):
        ax.axvline(0, color='k', linestyle='--')
        # Optional: Explicitly remove y-label text from the right plot if the 
        # custom plotting function re-adds it despite sharey=True
        if i > 0:
            ax.set_ylabel('')

    # --- LFP plot ---
    subfigs[1].suptitle('LFP', weight='bold')
    axs = subfigs[1].subplots(1, 2)
    pcolormesh_xr(lfp_mean, col_coord='time', row_coord='ch_y', 
                  subplot_coord='ch_shank', axs=axs)
    
    for i, ax in enumerate(axs.flatten()):
        ax.axvline(0, color='w', linestyle='--')
        if i > 0:
            ax.set_ylabel('')

    # --- CSD plot ---
    subfigs[2].suptitle('CSD', weight='bold')
    axs = subfigs[2].subplots(1, 2)
    pcolormesh_xr(csd_mean, col_coord='time', row_coord='ch_y', 
                  subplot_coord='ch_shank', axs=axs)
    
    for i, ax in enumerate(axs.flatten()):
        ax.axvline(0, color='w', linestyle='--')
        if i > 0:
            ax.set_ylabel('')

    return fig