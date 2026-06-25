# Histogram plotting functions
# Author: Drew Headley
# Date: 2024-06-17

import matplotlib.pyplot as plt
import numpy as np

def stairs_fl(values, edges, fill_color=None, edge_color=None,
              fill_alpha=0.5, edge_alpha=1.0, ax=None, **kwargs):
    """
    Plots a step function histogram with filled areas.

    Parameters
    ----------
    values : np.ndarray
        The heights of the steps.
    edges : np.ndarray
        The edges of the bins.
    fill_color : str or tuple, optional
        Color to fill the area under the steps (default is None).
    edge_color : str or tuple, optional
        Color of the step edges (default is None).
    fill_alpha : float, optional
        Alpha value for the filled area (default is 0.5).
    edge_alpha : float, optional
        Alpha value for the step edges (default is 1.0).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (default is None, creates a new figure).
    **kwargs : additional keyword arguments
        Additional parameters for `plt.plot`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes on which the histogram was plotted.
    """
    
    if ax is None:
        fig, ax = plt.subplots()

    if fill_color is not None:
        ax.stairs(values, edges, color=fill_color, alpha=fill_alpha, 
                  fill=True, **kwargs)

    if edge_color is not None:
        ax.stairs(values, edges, color=edge_color, alpha=edge_alpha,
                  fill=False, **kwargs)

    return ax