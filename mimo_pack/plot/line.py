# Plotting functions for line plots
# Drew Headley, 2025

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

def plot_mean_shaded_xr(x: xr.DataArray,
                     plot_dim: str,
                     ax=None,
                     err: str = 'sem',
                     ci: float = 95.0,
                     color: str = 'k',
                     alpha: float = 0.3,
                     label: str | None = None,
                     line_kwargs: dict | None = None,
                     fill_kwargs: dict | None = None):
    """
    Plot the mean of an xarray.DataArray along all dimensions except `plot_dim`,
    with shaded error bars.

    Parameters
    ----------
    x : xr.DataArray
        Input data.
    plot_dim : str
        Dimension to plot along (x-axis). Mean and error are computed over all
        other dimensions.
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, a new axes is created.
    err : {'sem','std','ci'}
        Error to display: standard error of the mean ('sem'), standard deviation ('std'),
        or confidence interval ('ci' uses normal approx and `ci` level).
    ci : float
        Confidence level (%) used when err == 'ci' (default 95).
    color : str
        Line / fill color.
    alpha : float
        Fill transparency.
    label : str or None
        Line label.
    line_kwargs : dict or None
        Additional kwargs passed to ax.plot for the mean line.
    fill_kwargs : dict or None
        Additional kwargs passed to ax.fill_between for shaded region.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.
    """
    if plot_dim not in x.dims:
        raise ValueError(f"plot_dim '{plot_dim}' not in x.dims: {x.dims}")

    reduce_dims = [d for d in x.dims if d != plot_dim]
    if len(reduce_dims) == 0:
        raise ValueError("No remaining dimensions to reduce over; nothing to average.")

    # compute mean
    mean = x.mean(dim=reduce_dims, skipna=True)

    # compute error band
    if err == 'std':
        err_arr = x.std(dim=reduce_dims, skipna=True)
        lower = mean - err_arr
        upper = mean + err_arr
    elif err == 'sem':
        std = x.std(dim=reduce_dims, skipna=True)
        # compute effective N along reduce_dims (product of counts)
        counts = x.count(dim=reduce_dims)
        # counts may be DataArray with same dims as mean (i.e. indexed by plot_dim)
        # Convert to float and avoid zeros
        counts = counts.where(counts > 0, other=np.nan).astype(float)
        sem = std / np.sqrt(counts)
        lower = mean - sem
        upper = mean + sem
    elif err == 'ci':
        std = x.std(dim=reduce_dims, skipna=True)
        counts = x.count(dim=reduce_dims)
        counts = counts.where(counts > 0, other=np.nan).astype(float)
        sem = std / np.sqrt(counts)
        z = norm.ppf(0.5 + ci/200.0)
        margin = z * sem
        lower = mean - margin
        upper = mean + margin
    else:
        raise ValueError("err must be one of {'sem','std','ci'}")

    # prepare plotting arrays
    x_vals = mean[plot_dim].values
    y_vals = mean.values
    y_lower = lower.values
    y_upper = upper.values

    if ax is None:
        fig, ax = plt.subplots()

    lkw = {} if line_kwargs is None else dict(line_kwargs)
    fkw = {} if fill_kwargs is None else dict(fill_kwargs)

    ax.plot(x_vals, y_vals, color=color, label=label, **lkw)
    ax.fill_between(x_vals, y_lower, y_upper, color=color, alpha=alpha, **fkw)

    if label is not None:
        ax.legend()

    return ax