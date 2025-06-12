# Mapped data plotting functions
# Author: Drew Headley
# Date: 2024-06-17

import matplotlib.pyplot as plt
import numpy as np

def wave_map(waves, x_pos, y_pos, c=None, x_scale=1, y_scale=1, 
             cmap='viridis', vmin=None, vmax=None, ax=None, 
             **kwargs):
    """
    Plots a map of waveforms across channels.

    Parameters
    ----------
    waves : N-length list of np.ndarray or (T, N) np.ndarray
        Waveforms for each channel. If a single array is provided,
        each column is treated as a channel waveform.
    x_pos : np.ndarray, (N,)
        The x coordinates of the channels.
    y_pos : np.ndarray, (N,)
        The y coordinates of the channels.
    c : np.ndarray, (N,) optional
        Values to use when setting the color of each waveform. If None,
        the waveforms are plotted black with no color scaling.
    x_scale : float, optional
        Scaling factor for x-axis (default is 1).
    y_scale : float, optional
        Scaling factor for y-axis (default is 1).
    cmap : str, optional,
        Colormap to use (default is 'viridis').
    vmin : float, optional
        Minimum value for color scaling (default is None).
    vmax : float, optional
        Maximum value for color scaling (default is None).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (default is None, creates a new figure).
    **kwargs : additional keyword arguments
        Additional parameters for `plt.plot`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes on which the waveforms were plotted.
    """
    
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(waves, np.ndarray):
        if waves.ndim == 1:
            waves = waves.reshape(-1, 1)
        elif waves.ndim == 2:
            waves = [waves[:, i] for i in range(waves.shape[1])]
    elif not isinstance(waves, list):
        raise ValueError("waves must be a list of np.ndarray or a 2D np.ndarray")

    if len(waves) != x_pos.size or len(waves) != y_pos.size:
        raise ValueError("Length of waves must match length of x_pos and y_pos")
    
    if c is None:
        c = np.repeat('black', len(waves))
    elif len(c) != len(waves):
        raise ValueError("Length of c must match length of waves")
    elif not isinstance(c, np.ndarray):
        c = np.array(c)

    # set colors by colormap
    if isinstance(c, np.ndarray) and np.issubdtype(c.dtype, np.number):
        if vmin is None:
            vmin = np.min(c)
        if vmax is None:
            vmax = np.max(c)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap)
        c = cmap(norm(c))
        

    for i, wave in enumerate(waves):
        if wave.ndim == 1:
            wave = wave.reshape(-1,1)

        x_vals = x_pos[i] + np.arange(wave.shape[0]) * x_scale
        y_vals = y_pos[i] + wave * y_scale

        ax.plot(x_vals, y_vals, color=c[i], **kwargs)

    return ax


def amp_map(amp, x_pos, y_pos, cmap='viridis', ax=None, **kwargs):
        """
        Plot a map of amplitude values across channels.

        Parameters
        ----------
        amp : np.ndarray, (N,)
            The amplitude values for each channel.
        x_pos : np.ndarray, (N,)
            The x coordinates of the channels.
        y_pos : np.ndarray, (N,)
            The y coordinates of the channels.
        cmap : str, optional
            Colormap to use for the plot, by default 'viridis'.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None a new figure is created, by default None.
        **kwargs : dict, optional
            Additional keyword arguments for the scatter plot.

        Returns
        -------
        ax: matplotlib.axes.Axes
            The axes with the amplitude map plotted.
        scat: matplotlib.collections.PathCollection
            The scatter plot object.
        """

        if ax is None:
            fig, ax = plt.subplots()
        
        # set point size to minimum non-zero distance between positions
        median_dist = (np.sqrt((x_pos.reshape(-1,1)-x_pos)**2 + (y_pos.reshape(-1,1)-y_pos)**2))
        median_dist = np.nanmin(median_dist[median_dist > 0])  # ignore zero distances

        scat = ax.scatter(x_pos, y_pos, s=median_dist,
                          c=amp, cmap=cmap, **kwargs)
                
        return ax, scat