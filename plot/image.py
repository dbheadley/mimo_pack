# Plotting functions for image data
# Author: Drew Headley

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def composite_image(data, colors=None, labels=None, background=None, 
                                   colorbars=True, ax=None, **kwargs):
    """
    Create an alpha-blended RGB composite image from up to 3 data arrays, 
    each mapped to a specified color.
    
    Args:
        data: 3D numpy arrays
            A 3D numpy array to plot. Each array along the third dimension will 
            be assigned a color from the `colors` array.
        colors: Nx3 array-like
            Each row is an RGB color (0-1 or 0-255) for each data array.
        labels: list of strings
            A list of labels for each data array (used for the color legend).
        background: 3-length array-like
            RGB color (0-1 or 0-255) for the background (default is white).

    Optional
    --------
        colorbars: bool
            Whether to add colorbars for each data array (default is True).
        ax: Matplotlib axis object 
            Axes to draw the composite image on (if not provided, a new figure is created).
    Returns:
        ax: Matplotlib axis object containing the composite image
        composite: RGB image as numpy array, dtype float32, shape (H, W, 3)
    """

    n = data.shape[2]
    if colors is None:
        def_map = plt.get_cmap('tab10')
        colors = def_map(np.mod(np.arange(n), 10))[:,:3]  # Get first n colors from the colormap
    else:
        colors = np.array(colors)
        if colors.shape[0] < n:
            raise ValueError("Not enough colors specified for the number of data arrays.")
    
    if labels is None:
        labels = [f"Data {i}" for i in range(n)]

    if ax is None:
        fig, ax = plt.subplots()

    data_mins = np.nanmin(data, axis=(0,1), keepdims=True)
    data_maxs = np.nanmax(data, axis=(0,1), keepdims=True)

    # Normalize data to [0,1] for each channel
    data = np.clip((data - data_mins) / (data_maxs - data_mins + 1e-8), 0, 1)

    # Prepare background
    shape = data.shape[:2]
    if background is None:
        back_color = np.ones((1,1,3))
    else:
        try:
            back_color = np.array(background).reshape(1, 1, 3).astype(np.float32)
        except:
            raise ValueError("Background color must be 3 length vector")

    # Alpha blending
    alpha = np.max(data,axis=2,keepdims=True)

    # Create image using additive color mixing
    add_img = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    for i in range(n):
        add_img += data[:,:,i][:,:,np.newaxis] * colors[i].reshape(1, 1, 3)

    composite = (1 - alpha) * back_color + alpha * add_img
    composite /= np.max(composite)
    ax.imshow(composite, **kwargs)

    # add a colorbar for each data array
    if colorbars:
        for i in range(n):
            cmap = LinearSegmentedColormap.from_list("custom_cmap", [back_color.reshape(-1), colors[i]])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=data_mins[0,0,i], vmax=data_maxs[0,0,i]))
            sm.set_array([])
            ax.figure.colorbar(sm, ax=ax, orientation='vertical', label=labels[i])

    return ax, composite