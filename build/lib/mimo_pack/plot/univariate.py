# Univariate plotting functions
# Author: Drew Headley
# Date: 2024-06-17

import matplotlib.pyplot as plt
import numpy as np

def violinplot_log(dataset, axes=None, **kwargs):
    """
    Create a violin plot of the dataset with a log scale on the y-axis.
    
    Parameters
    ----------
    dataset : array-like
        The dataset to plot.
    **kwargs : dict
        Additional keyword arguments to pass to the violinplot function.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """

    # create a new figure if axes are not provided
    if axes is None:
        fig, ax = plt.subplots()
    else:
        ax = axes

    # log transfom the data
    # get range of data values, set min and max to nearest power of 10
    if type(dataset) == list:
        dataset = [np.log10(x) for x in dataset]
        data_min = np.floor(np.min([np.min(x) for x in dataset]))
        data_max = np.ceil(np.max([np.max(x) for x in dataset]))
    else:
        dataset = np.log10(dataset)
        data_min = np.floor(np.min(dataset))
        data_max = np.ceil(np.max(dataset))
    
    # create a log scale for the y-axis
    pow10 = np.power(10, np.arange(data_min, data_max+1))
    yticks = (np.arange(1, 10) * pow10[:, np.newaxis]).ravel()

    # create the plot
    ax.violinplot(dataset, **kwargs)

    # set the y-axis scale and labels
    ax.set_yticks(np.log10(yticks))

    # label the y-ticks that are in pow10
    yticklabels = [f'{float(y)}' if y in pow10 else '' for y in yticks]
    ax.set_yticklabels(yticklabels)

    return ax