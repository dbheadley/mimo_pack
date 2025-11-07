# Helper functions for working with xarray objects
# Author: Drew Headley
# Date: 2025-10-06

import xarray as xr
import numpy as np
import pandas as pd

def window_xr(data: xr.DataArray, dim: str, centers: np.ndarray, 
              pre: float|int, post: float|int, indices: bool = False) -> xr.DataArray:
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

def window_df_xr(data: xr.DataArray, df: pd.DataFrame|pd.Series, 
                 center_cols: str,  dim: str, pre: float|int, post: float|int, 
                 coord_cols: str|list = None, **kwargs) -> xr.DataArray:
    """
    Apply a windowing function to an xarray DataArray based on a pandas DataFrame or Series.

    Parameters
    ----------
    data : xr.DataArray
        The xarray DataArray to window.
    df : pd.DataFrame|pd.Series
        The pandas DataFrame or Series containing the windowing information.
    center_cols : str
        The column(s) in df that contain the center points for the windows.
    coord_cols : str|list, optional
        The column(s) in df that contain the coordinates for the windows.
    dim : str
        The dimension along which to window (e.g., 'time').
    pre : float|int
        A float specifying the length before each center.
    post : float|int
        A float specifying the length after each center.
    **kwargs
        Additional keyword arguments to pass to the windowing function.

    Returns
    -------
    out_xr : xr.DataArray
        A new xarray DataArray containing the windowed data.
    """

    centers = df[center_cols].values
    if coord_cols is not None:
        coords = df[coord_cols].values
    else:
        coords = None

    out_xr = window_xr(data, dim=dim, centers=centers, pre=pre, post=post, **kwargs)

    if coords is not None:
        for i, col in enumerate(coord_cols):
            out_xr = out_xr.assign_coords({col: (('window',), coords[:, i])})

    return out_xr

def zscore_xr(x: xr.DataArray, dim: str|list = None, bg: dict = None,
              skipna = True) -> xr.DataArray:
    """
    Z-score the data based on background mean and std.

    Parameters
    ----------
    x : xr.DataArray
        The input data array to be z-scored.
    dim : str|list, optional
        The dimensions along which to z-score. If None, z-scores
        along all dimensions. Default is None.
    bg : dict
        Dictionary containing data range used for calculating
        background mean and standard deviation. Should have keys
        for xarray dimensions and specify the slice for each dimension.
        E.g. {'time': slice(-1, 0)}.
        Default is all data.
    skipna : bool, optional
        Whether to skip NaN values when calculating the z-score.
        Default is True.

    Returns
    -------
    xz :xr.DataArray
        The z-scored data array.
    """

    if bg is None:
        bg = x
    else:
        bg = x.sel(bg)
    
    mean = bg.mean(dim=dim, skipna=skipna)
    std = bg.std(dim=dim, skipna=skipna)

    xz = (x - mean) / std
    return xz