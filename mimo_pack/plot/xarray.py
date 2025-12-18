import numpy as np
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

def _plot_xr_common(data, plot_func, subplot_coord, row_coord, col_coord, 
                    extra_coord=None, axs=None):
    """
    Generic driver for plotting xarray data on a grid of subplots.
    Handles grouping, axis creation, dimension resolution, and standard labeling.
    """
    # 1. Group Data
    if subplot_coord is None:
        groups = [(None, data)]
    elif subplot_coord in data.coords:
        groups = list(data.groupby(subplot_coord))
    else:
        # Fallback if coord string provided but missing
        groups = [('All', data)]
        
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

    last_plot_obj = None

    # 3. Iterate and Plot
    for i, (key, sub_da) in enumerate(groups):
        ax = axs[i]
        
        # Handle formatting for 'None' subplot_coord (single plot case)
        if subplot_coord is None:
            title_prefix = ""
            key_str = ""
        else:
            title_prefix = f"{subplot_coord} "
            key_str = str(key)

        if sub_da.size == 0:
            ax.set_title(f'{title_prefix}{key_str}\n(empty)'.strip())
            continue

        # Resolve dimensions
        try:
            row_dim = _get_dim(sub_da, row_coord)
            col_dim = _get_dim(sub_da, col_coord)
            plot_dims = [row_dim, col_dim]
            
            if extra_coord:
                extra_dim = _get_dim(sub_da, extra_coord)
                plot_dims.append(extra_dim)
        except ValueError:
            ax.set_title(f'{title_prefix}{key_str}\n(missing coords)'.strip())
            continue

        # Validate and Squeeze extra dimensions
        expected_dims = set(plot_dims)
        current_dims = set(sub_da.dims)
        extra_dims = current_dims - expected_dims

        if extra_dims:
            non_singleton = [d for d in extra_dims if sub_da.sizes[d] > 1]
            if non_singleton:
                raise ValueError(
                    f"Data contains non-singleton extra dimensions {non_singleton}. "
                    f"Resolved plotting dims: {plot_dims}."
                )
            sub_da = sub_da.squeeze(dim=list(extra_dims))

        # Transpose to standard order
        try:
            plot_data = sub_da.transpose(*plot_dims)
        except ValueError as e:
            raise ValueError(f"Error transposing dimensions: {e}")

        # Execute specific plotting logic
        # plot_func signature: (ax, data_array)
        last_plot_obj = plot_func(ax, plot_data)

        # Labels and Formatting
        if subplot_coord is not None:
            ax.set_title(f'{subplot_coord} {key}')
            
        ax.set_xlabel(col_coord)
        
        if i == 0:
            ax.set_ylabel(row_coord)
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])

    return fig, axs, last_plot_obj

def composite_xr(data, subplot_coord='ch_shank', color_coord='ch_x', 
                 row_coord='ch_y', col_coord='time', 
                 background=[0, 0, 0], axs=None):
    """
    Plot composite images of pre-processed data grouped by a coordinate.
    """
    
    def _render(ax, da):
        img = da.values
        # Flip y-axis (rows) so higher coordinates appear at the top
        img = np.flip(img, axis=0)
        
        # Extract extents
        col_vals = da.coords[col_coord].values
        row_vals = da.coords[row_coord].values
        extent = [col_vals[0], col_vals[-1], row_vals[0], row_vals[-1]]
        
        composite_image(
            img,
            extent=extent,
            ax=ax,
            colorbars=False,
            aspect='auto',
            background=background
        )
        return None # No mappable to return for composite

    fig, axs, _ = _plot_xr_common(
        data, 
        _render, 
        subplot_coord, 
        row_coord, 
        col_coord, 
        extra_coord=color_coord, 
        axs=axs
    )
    return fig, axs

def pcolormesh_xr(data, subplot_coord='ch_shank', row_coord='ch_y', col_coord='time', 
                  vmin=None, vmax=None, cmap='viridis', axs=None):
    """
    Plot pcolormesh heatmaps of data.
    """
    
    def _render(ax, da):
        return ax.pcolormesh(
            da.coords[col_coord].values,
            da.coords[row_coord].values,
            da.values,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            shading='auto'
        )

    fig, axs, mesh = _plot_xr_common(
        data, 
        _render, 
        subplot_coord, 
        row_coord, 
        col_coord, 
        extra_coord=None, 
        axs=axs
    )
        
    return fig, axs