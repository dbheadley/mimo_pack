import pandas as pd

# Plotting functions for video and related date

import matplotlib.pyplot as plt

def plot_video_data(tbl_path, ax=None, **kwargs):
    """Plot video segments as rectangles on a Matplotlib axis.
    This function reads video segment data from a specified CSV file. It
    identifies segments based on the 'epoch_num' column and plots their
    duration, determined by the 'time' column, as rectangles on a given or
    newly created Matplotlib axis.
    Parameters
    ----------
    tbl_path : str
        Path to the CSV file with video data. The file must contain
        'epoch_num' and 'time' columns.
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If None, a new figure and axis are created.
    **kwargs
        Arbitrary keyword arguments passed to `matplotlib.patches.Rectangle`
        to control the properties of the plotted segments (e.g., color, alpha).
    Returns
    -------
    matplotlib.axes.Axes
        The axis with the plotted video segments.
    """
    
    # confirm that highspeed camera segments are identified that were aligned with neural data
    vid_df = pd.read_csv(tbl_path)

    # display video data
    segs_df = vid_df.groupby('epoch_num')['time'].agg(['min', 'max']).reset_index()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2))

    # Set default patch properties, which can be overridden by kwargs
    patch_defaults = {'color': 'red', 'alpha': 0.5}
    patch_defaults.update(kwargs)

    # Plot rectangles for each video segment
    for start, end in zip(segs_df['min'], segs_df['max']):
        ax.add_patch(plt.Rectangle((start, 0), end - start, 1, **patch_defaults))

    ax.set_xlim(0, vid_df['time'].max())
    ax.set_ylim(-1, 2)
    
    return ax