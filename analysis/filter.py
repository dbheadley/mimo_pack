# Functions for filtering
# Author: Drew Headley
# Date: 2025-07-22

import xarray as xr
from scipy.signal import butter, filtfilt

def lp_filter_xr(signal_xr, cutoff_freq=10, order=4):
    """
    Applies a low-pass Butterworth filter to the input signal.

    Parameters
    ----------
    signal_xr : xr.DataArray
        The input signal to be filtered.
    cutoff_freq : float
        The cutoff frequency for the low-pass filter in Hz.
    order : int, optional
        The order of the Butterworth filter. Default is 4.

    Returns
    -------
    xr.DataArray
        The filtered signal.
    """

    # if signal_xr doesn't have sample_rate attribute, calculate from time dimension
    if 'sample_rate' not in signal_xr.attrs:
        s_rate = 1/signal_xr.time.diff('time').mean().item()
    else:
        s_rate = signal_xr.attrs.get('sample_rate')
        
    nyquist = 0.5 * s_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    dims = signal_xr.dims
    time_idx = dims.index('time')
    
    signal_filt = filtfilt(b,a, signal_xr.to_numpy(), axis=time_idx)

    return xr.DataArray(
        signal_filt,
        dims=dims,
        coords=signal_xr.coords,
        attrs=signal_xr.attrs
    )

def hp_filter_xr(signal_xr, cutoff_freq=0.02, order=4):
    """
    Applies a high-pass Butterworth filter to the input signal.

    Parameters
    ----------
    signal_xr : xr.DataArray
        The input signal to be filtered.
    cutoff_freq : float
        The cutoff frequency for the high-pass filter in Hz.
    order : int, optional
        The order of the Butterworth filter. Default is 4.

    Returns
    -------
    xr.DataArray
        The filtered signal.
    """
    s_rate = signal_xr.attrs.get('sample_rate', 1.0)
    nyquist = 0.5 * s_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    
    dims = signal_xr.dims
    time_idx = dims.index('time')
    
    signal_filt = filtfilt(b,a, signal_xr.to_numpy(), axis=time_idx)

    return xr.DataArray(
        signal_filt,
        dims=dims,
        coords=signal_xr.coords,
        attrs=signal_xr.attrs
    )