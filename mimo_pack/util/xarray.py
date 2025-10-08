# Helper functions for working with xarray objects
# Author: Drew Headley
# Date: 2025-10-06

import xarray as xr
import numpy as np

def window_xr(data: xr.DataArray, dim: str, centers: np.ndarray, 
              pre: float, post: float, indices: bool = False) -> xr.DataArray:
    """
    Windows an xarray DataArray along a specified dimension.

    This function takes an xarray DataArray and extracts windows of data
    centered around specified points along a given dimension. The window
    dimension should be a continuous variable linearly spaced (e.g., time).

    Parameters
    ----------
    data : xr.DataArray
        The xarray DataArray to window.
    dim : str
        The dimension along which to window (e.g., 'time').
    centers : np.ndarray
        An array of center points for the windows.
    pre : float
        A float specifying the length before each center.
    post : float
        A float specifying the length after each center.
    indices : bool, optional
        If True, the centers, pre, and post are treated as indices rather 
        than coordinate values. If false, indices are calculated. Default is False.

    Returns
    -------
    xr.DataArray
        A new xarray DataArray containing the windowed data, concatenated
        along a new dimension called 'window'. The windowed dimension is
        replaced with a new dimension with the suffix '_relative' indicating
        the relative position within each window.
    """

    windowed_data = []
    dim_values = data[dim].values
    dim_step = np.mean(np.diff(dim_values))
    
    # If indices is False, convert window centers and edges to indices
    if not indices:
        centers = np.array([np.abs(dim_values - c).argmin() for c in centers])
        pre = np.round(pre / dim_step)
        post = np.round(post / dim_step)


    relative_coords = np.arange(-pre, post) * dim_step

    for i, center in enumerate(centers):
        start = int(center - pre)
        end = int(center + post)

        if start < 0 or end > dim_values.size:
            raise ValueError(f"Window {i} is out of bounds.")
        
        window_slice = data.isel({dim: slice(start, end)})
        
        # Create a new coordinate for the relative position within the window
        windowed_data.append(window_slice)
        # set the dim dimension to relative coords
        windowed_data[-1] = windowed_data[-1].assign_coords({dim: relative_coords})

    # Concatenate all windowed slices along a new dimension 'window'
    out_xr = xr.concat(windowed_data, dim='window')
    out_xr = out_xr.assign_coords({'window': np.arange(len(centers))})
    
    # rename the dim dimension to indicate it's relative
    out_xr = out_xr.rename({dim: f"{dim}_relative"})

    return out_xr