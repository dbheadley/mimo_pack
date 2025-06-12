# Time series plotting functions
# Author: Drew Headley
# Date: 2024-06-17

import matplotlib.pyplot as plt
import numpy as np
def stacked_lines(data_vals, x_vals=None, disp_color='black', 
                       sep_ratio=1, labels=None, labels_right=None, 
                       abs_sep=None, ax=None, **kwargs):
    """
    Plots a series of line plots stacked along the y-axis.

    Parameters
    ----------
    data_vals : ndarray
        An NxM array, where each column is a different line to be plotted.
    x_vals : array-like, optional
        An array specifying the x-coordinates for each column in `data_vals`.
        If None, the indices of the columns are used.
    disp_color : str, list, or ndarray, optional
        The color of each line. Can be a single color (applied to all lines),
        a list of color names, or an Nx3 array of RGB values.
    sep_ratio : float, optional
        The separation ratio between lines based on their maximum value. Default is 1.
    labels : list of str, optional
        Labels for each line, displayed on the left side.
    labels_right : list of str, optional
        Labels for each line, displayed on the right side.
    abs_sep : float, optional
        Absolute separation between lines. Overrides `sep_ratio` if provided.
    **kwargs : dict
        Additional keyword arguments passed to the `plot` function.

    Returns
    -------
    ax : axes object
        The axes object containing the plot.
    """
    # Validate inputs
    if x_vals is None:
        x_vals = np.arange(data_vals.shape[0])
    if len(x_vals) != data_vals.shape[0]:
        raise ValueError("Length of x_vals must match the number of rows in data_vals.")
    
    # Handle disp_color
    if isinstance(disp_color, str):
        disp_color = [disp_color] * data_vals.shape[1]
    elif isinstance(disp_color, list):
        disp_color = [plt.colors.to_rgb(c) if isinstance(c, str) else c for c in disp_color]
    disp_color = np.array(disp_color)


    if disp_color.shape[0] == 1:
        disp_color = np.tile(disp_color, (data_vals.shape[1], 1))
    elif disp_color.shape[0] != data_vals.shape[1]:
        raise ValueError("Number of colors must match the number of rows in data_vals.")

    if ax is None:
        fig, ax = plt.subplots()

    # Handle separation
    if abs_sep is not None:
        line_sep = abs_sep
    else:
        max_diff = np.max(np.abs(np.max(data_vals, axis=0) - np.min(data_vals, axis=0)))
        line_sep = max_diff * sep_ratio

    # Plot lines
    line_handles = []
    for i, row in enumerate(data_vals.T):
        offset = i * line_sep
        line, = plt.plot(x_vals, row + offset, color=disp_color[i], **kwargs)
        line_handles.append(line)
        plt.axhline(y=offset, color='gray', linestyle=':', linewidth=0.5)
        if labels:
            plt.text(x_vals[0], offset, labels[i], color=disp_color[i], verticalalignment='center')
        if labels_right:
            plt.text(x_vals[-1], offset, labels_right[i], color=disp_color[i], verticalalignment='center')

    return ax

# Example usage
if __name__ == "__main__":
    num_trials = 10
    dur_trial = 100
    fake_data = np.random.rand(num_trials, dur_trial)
    labels = [f"Line {i+1}" for i in range(num_trials)]
    stacked_lines(fake_data, disp_color='blue', sep_ratio=2, labels=labels)
    plt.show()