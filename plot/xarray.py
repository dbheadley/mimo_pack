import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mimo_pack.plot.image import composite_image

def _get_dim(da, coord_name):
    """Helper to resolve a coordinate name to its underlying dimension."""
    if coord_name in da.dims:
        return coord_name
    elif coord_name in da.coords:
        return da.coords[coord_name].dims[0]
    else:
        raise ValueError(f"Coordinate '{coord_name}' not found in DataArray.")

def _prepare_data(data, row_coord, col_coord, extra_coord=None):
    """
    Common logic to resolve dimensions, squeeze extra dims, and transpose 
    xarray data for plotting. Handles grouping for extra_coord stacking.
    """

    row_dim = _get_dim(data, row_coord)
    col_dim = _get_dim(data, col_coord)


    # 1. Squeeze Extra Dimensions
    # Identify needed dims
    needed_dims = {row_dim, col_dim}
    if extra_coord and extra_coord in data.dims:
        needed_dims.add(extra_coord)
    elif extra_coord and extra_coord in data.coords:
        needed_dims.add(data.coords[extra_coord].dims[0])
    
    current_dims = set(data.dims)
    squeeze_dims = [d for d in current_dims if d not in needed_dims and data.sizes[d] > 1]
    
    if squeeze_dims:
        raise ValueError(
            f"Data contains non-singleton extra dimensions {squeeze_dims}. "
            f"Resolved plotting dims: {needed_dims}."
        )
    
    data = data.squeeze() 

    # 2. Handle extra_coord (Stacking)
    if extra_coord:
        if extra_coord not in data.coords and extra_coord not in data.dims:
             raise ValueError(f"Coordinate '{extra_coord}' not found in DataArray.")
        
        try:
            slices = []
            keys = []
            
            for k, g in data.groupby(extra_coord):
                # Transpose to (row, col)
                g_trans = g.transpose(row_dim, col_dim)
                
                # remove coordinate to avoid duplication on concat
                g_trans = g_trans.drop_vars(extra_coord, errors='ignore')
                slices.append(g_trans)
                keys.append(k)
            
            # Concatenate along new dimension corresponding to extra_coord
            idx = pd.Index(keys, name=extra_coord)
            plot_data = xr.concat(slices, dim=idx)
            
            # Final Transpose: (row, col, extra)
            plot_data = plot_data.transpose(row_dim, col_dim, extra_coord)
            
            return plot_data
        
        except Exception as e:
            raise ValueError(f"Error stacking data by '{extra_coord}': {e}")

    else:
        # Standard Transpose
        try:
            plot_data = data.transpose(row_dim, col_dim)
            return plot_data
        except Exception as e:
             raise ValueError(f"Error transposing to ({row_coord}, {col_coord}): {e}")

def _apply_subplots(data, plot_func, subplot_coord, axs=None, **kwargs):
    """
    Generic driver for applying a single-plot function across groups defined by subplot_coord.
    """
    # 1. Group Data
    if subplot_coord is None:
        # Treat as single group if None passed
        groups = [(None, data)]
    elif subplot_coord in data.coords:
        groups = list(data.groupby(subplot_coord))
    else:
        raise ValueError(f"Coordinate '{subplot_coord}' not found in DataArray.")
        
    n_plots = len(groups)

    # 2. Setup Axes
    if axs is None:
        fig, axs = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), squeeze=False)
        axs = axs.flatten()
    else:
        fig = plt.gcf()
        axs = np.atleast_1d(axs)
        if len(axs) < n_plots:
             raise ValueError(f"Provided {len(axs)} axes, but data requires {n_plots}.")

    last_return = None

    # 3. Iterate and Plot
    for i, (key, sub_da) in enumerate(groups):
        ax = axs[i]
        
        if sub_da.size == 0:
            ax.set_title(f'{subplot_coord} {key}\n(empty)')
            continue

        # Execute specific plotting function
        last_return = plot_func(sub_da, ax=ax, **kwargs)

        # Labels and Formatting
        ax.set_title(f'{subplot_coord} {key}')
            
        if i > 0:
            ax.set_ylabel('')
            # Optional: remove y-ticks for cleaner grid?
            # ax.set_yticklabels([]) 

    return fig, axs, last_return

def composite_xr(data, color_coord='ch_x', row_coord='ch_y', col_coord='time', 
                 background=[0, 0, 0], ax=None):
    """
    Refomats xarray data and produces a composite image plot on a single axes.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Prepare Data
    try:
        plot_data = _prepare_data(data, row_coord, col_coord, color_coord)
    except ValueError as e:
        ax.text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center', transform=ax.transAxes)
        return fig, ax

    img = plot_data.values
    # Flip y-axis (rows) so higher coordinates appear at the top
    img = np.flip(img, axis=0)
    
    # Extract extents
    col_vals = plot_data.coords[col_coord].values
    row_vals = plot_data.coords[row_coord].values
    extent = [col_vals[0], col_vals[-1], row_vals[0], row_vals[-1]]
    
    composite_image(
        img,
        extent=extent,
        ax=ax,
        colorbars=False,
        aspect='auto',
        background=background
    )
    
    ax.set_xlabel(col_coord)
    ax.set_ylabel(row_coord)
    
    return fig, ax

def composite_xr_subplots(data, subplot_coord='ch_shank', color_coord='ch_x', 
                 row_coord='ch_y', col_coord='time', 
                 background=[0, 0, 0], axs=None):
    """
    Plot composite images of data grouped by subplot_coord.
    """
    return _apply_subplots(
        data, 
        composite_xr, 
        subplot_coord, 
        axs=axs,
        color_coord=color_coord,
        row_coord=row_coord,
        col_coord=col_coord,
        background=background
    )

def pcolormesh_xr(data, row_coord='ch_y', col_coord='time', 
                  vmin=None, vmax=None, cmap='viridis', ax=None):
    """
    Reformats xarray data and produces a pcolormesh plot on a single axes.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Prepare Data
    try:
        plot_data = _prepare_data(data, row_coord, col_coord)
    except ValueError as e:
        ax.text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center', transform=ax.transAxes)
        return fig, ax, None

    mesh = ax.pcolormesh(
        plot_data.coords[col_coord].values,
        plot_data.coords[row_coord].values,
        plot_data.values,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        shading='auto'
    )
    
    ax.set_xlabel(col_coord)
    ax.set_ylabel(row_coord)

    return fig, ax, mesh

def pcolormesh_xr_subplots(data, subplot_coord='ch_shank', row_coord='ch_y', col_coord='time', 
                  vmin=None, vmax=None, cmap='viridis', axs=None):
    """
    Plot pcolormesh heatmaps of data grouped by subplot_coord.
    """
    return _apply_subplots(
        data,
        pcolormesh_xr,
        subplot_coord,
        axs=axs,
        row_coord=row_coord,
        col_coord=col_coord,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap
    )