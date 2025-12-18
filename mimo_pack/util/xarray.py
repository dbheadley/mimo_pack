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


def interp_stack_xr(da: xr.DataArray, coord_interp: str ='ch_y', coord_stack: str ='ch_x',
                    spacing: float =15, coord_min: float =None, coord_max: float =None):
    """
    Interpolate voltages along one axis and stack along the other from an
    xarray.DataArray onto a regular spatial grid.

    Iterates over the stacking coordinate using xarray.groupby, interpolates
    the data within each group onto a regular grid, and stacks the results.
    Preserves coordinates associated with the stacking dimension if they are
    consistent within groups, and preserves all coordinates associated with
    non-interpolated dimensions.

    Parameters
    ----------
    da : xarray.DataArray
        Input data. Must have coordinates specified by coord_interp and 
        coord_stack.
    coord_interp : str, optional
        Name of the coordinate to interpolate along. Default is 'ch_y'.
    coord_stack : str, optional
        Name of the coordinate to stack groups along. Default is 'ch_x'.
    spacing : float, optional
        Spatial spacing for the interpolation grid. Default is 15.
    coord_min : float, optional
        Minimum value for the interpolation grid. If None, uses the 
        minimum of the coord_interp values.
    coord_max : float, optional
        Maximum value for the interpolation grid. If None, uses the 
        maximum of the coord_interp values.

    Returns
    -------
    interp_da : xarray.DataArray
        Interpolated data with new spatial dimensions replacing the 
        original channel dimension.
    """

    if coord_interp not in da.coords or coord_stack not in da.coords:
        raise ValueError(f"DataArray must contain coords '{coord_interp}' and '{coord_stack}'.")

    # Ensure both coord_interp and coord_stack live on the same source dimension
    interp_dims = da.coords[coord_interp].dims
    stack_dims = da.coords[coord_stack].dims
    
    if len(interp_dims) != 1 or len(stack_dims) != 1:
        raise ValueError("coord_interp and coord_stack must be 1-D coordinates.")
    
    channel_dim = interp_dims[0]
    if stack_dims[0] != channel_dim:
        raise ValueError(f"'{coord_interp}' and '{coord_stack}' must refer to the same dimension.")

    # Identify other dimensions (e.g. time, trials)
    orig_dims = list(da.dims)
    channel_pos = orig_dims.index(channel_dim)
    other_dims = [d for d in orig_dims if d != channel_dim]

    # Identify candidate coordinates to preserve along the stack dimension
    # These are coords that depend on channel_dim but are not the interp/stack coords themselves
    stack_cand_coords = [
        c for c in da.coords 
        if channel_dim in da.coords[c].dims 
        and c not in [coord_interp, coord_stack]
    ]
    
    # Dictionary to collect values for stack candidate coords: {name: [val_group1, val_group2, ...]}
    stack_coords_vals = {c: [] for c in stack_cand_coords}
    
    # Track which candidates remain valid (constant within every group so far)
    valid_stack_coords = set(stack_cand_coords)

    # Define interpolation grid
    amin = float(da.coords[coord_interp].min().values) if coord_min is None else float(coord_min)
    amax = float(da.coords[coord_interp].max().values) if coord_max is None else float(coord_max)
    interp_grid = np.arange(amin, amax + spacing, spacing)
    n_interp = interp_grid.size

    interp_groups = []
    group_keys = []

    # Iterate over the stacking coordinate using groupby
    for key, group in da.groupby(coord_stack):
        
        # Extract interpolation coordinates for this group
        coords = group[coord_interp].values
        
        # Skip groups with fewer than 2 points (cannot interpolate)
        if np.unique(coords).size < 2:
            continue

        # --- Handle Stack Coordinate Preservation ---
        # Check if candidate coordinates are constant in this group
        for c in list(valid_stack_coords):
            # We use the raw values from the group
            c_vals = group[c].values
            unique_vals = np.unique(c_vals)
            
            # If strictly one unique value, we can potentially preserve it
            if unique_vals.size == 1:
                stack_coords_vals[c].append(unique_vals[0])
            else:
                # Variable within group; cannot be a stack coordinate
                valid_stack_coords.remove(c)
                del stack_coords_vals[c]

        # --- Interpolation ---
        # Prepare data: Move channel_dim to front (axis 0)
        x = group.transpose(channel_dim, *other_dims)
        
        # Get shapes for reshaping
        channel_len = x.sizes[channel_dim]
        other_shapes = tuple(x.sizes[d] for d in other_dims)
        n_slices = int(np.prod(other_shapes)) if other_shapes else 1

        # Convert to numpy and collapse trailing dims: (n_chan_in_group, n_slices)
        vals = x.values.reshape((channel_len, n_slices))

        # Sort based on spatial coordinate
        order = np.argsort(coords)
        coords_sorted = coords[order]
        vals_sorted = vals[order, :]

        # Interpolate each collapsed slice
        # Output shape: (n_interp, n_slices)
        grp_res = np.empty((n_interp, n_slices), dtype=vals.dtype)
        
        for j in range(n_slices):
            grp_res[:, j] = np.interp(interp_grid, coords_sorted, vals_sorted[:, j])

        # Reshape back to (n_interp, *other_shapes)
        if other_shapes:
            grp_res = grp_res.reshape((n_interp,) + other_shapes)
        else:
            grp_res = grp_res.reshape((n_interp,))

        interp_groups.append(grp_res)
        group_keys.append(key)

    if not interp_groups:
        raise ValueError("No valid groups found for interpolation.")

    # Stack groups: (coord_interp, coord_stack, *other_dims)
    stacked = np.stack(interp_groups, axis=1)

    # Reorder axes to match original dimensions
    current_axes = [coord_interp, coord_stack] + other_dims
    
    # Final structure replaces channel_dim with [coord_interp, coord_stack]
    final_dims = orig_dims.copy()
    final_dims[channel_pos:channel_pos+1] = [coord_interp, coord_stack]

    perm = [current_axes.index(d) for d in final_dims]
    arr_permuted = np.transpose(stacked, perm)

    # --- Build Output Coordinates ---
    coords_dict = {}

    # 1. New Interpolation and Stack Coordinates
    coords_dict[coord_interp] = interp_grid
    coords_dict[coord_stack] = np.array(group_keys)

    # 2. Preserved Stack Coordinates (Auxiliary info on the stack dimension)
    for c in valid_stack_coords:
        # These now live along the coord_stack dimension
        coords_dict[c] = (coord_stack, np.array(stack_coords_vals[c]))

    # 3. Preserved Coordinates from Other Dimensions (e.g. Time)
    # Copy any coordinate that does NOT depend on the channel_dim we just removed
    for c in da.coords:
        if channel_dim not in da.coords[c].dims:
            coords_dict[c] = da.coords[c]

    interp_da = xr.DataArray(arr_permuted, dims=tuple(final_dims), coords=coords_dict,
                             attrs=da.attrs)

    return interp_da