# Signal processing (not filtering)

import numpy as np
import xarray as xr
from scipy.signal import hilbert

def envelope_xr(signal, dim='time'):
    """
    Calculate the amplitude envelope of a signal using the Hilbert transform.

    Parameters
    ----------
    signal : xarray.DataArray
        Input signal. Must have the dimension specified by 'dim'.
    dim : str, optional
        Dimension along which to calculate the envelope. Default is 'time'.

    Returns
    -------
    xarray.DataArray
        Amplitude envelope with the same dimensions and coordinates as the input.
    """

    if dim not in signal.dims:
        raise ValueError(f"Input xarray must have a '{dim}' dimension.")

    # Extract data and dimension info
    arr = signal.to_numpy()
    dims = signal.dims
    
    # Find the axis index for the specified dimension
    axis = np.where(np.array(dims) == dim)[0][0]

    # Calculate analytic signal using Hilbert transform
    # Note: scipy.signal.hilbert calculates the analytic signal
    analytic = hilbert(arr, axis=axis)
    
    # Calculate amplitude envelope
    env = np.abs(analytic)

    # Reconstruct xarray
    # Since dimensions and shape are preserved, we can copy coordinates directly
    return xr.DataArray(env, dims=dims, coords=signal.coords, attrs=signal.attrs)