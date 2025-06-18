# Functions for spectral analysis
# Author: Drew Headley
# Date: 2025-06-16

import xarray as xr
import numpy as np
import scipy.signal as ss
from mimo_pack.math.curvefitting import fit_exp_knee, exp_knee

def stft_xr(signal, window=1, **kwargs):
    """
    Apply Short-Time Fourier Transform (STFT) to an xarray signal and return an xarray with a frequency dimension.

    Parameters
    ----------
    signal : xarray.DataArray
        1D or 2D xarray signal. Must have a 'time' dimension.
    window : float
        Duration (in seconds) of each STFT window. Default is 1 second.
    kwargs : dict, optional
        Additional arguments to pass to scipy.signal.stft (e.g., nperseg, noverlap, etc.).

    Returns
    -------
    xarray.DataArray
        STFT result with same non-time dimensions as input, 'frequency' and new 'stft_time' dimensions.
    """
    if not isinstance(window, np.ndarray):
        window = np.array([window])

    # Extract time and sampling frequency
    time = signal['time'].values
    dt = np.nanmedian(np.diff(time))
    fs = 1.0 / dt

    arr = signal.values
    if arr.ndim == 1:
        arr = arr[:, None]

    hop = int(window * fs)
    mfft = kwargs.get('mfft', int(window * fs))
    kwargs['mfft'] = mfft

    win_func = ss.windows.gaussian(mfft, std=mfft/5.0, sym=True)
    stft = ss.ShortTimeFFT(win_func, hop, fs, **kwargs)
    
    f = stft.f
    t = stft.t(arr.shape[0])
    #Zxx = np.zeros((f.size, t.size, arr.shape[1]))
    Zxx = stft.spectrogram(arr, axis=0)
        
    # rearrange spectrogram array to have time, channel, frequency organization
    Zxx = np.transpose(Zxx, [2, 1, 0])

    # Prepare output dims and coords
    dims = list(signal.dims)
    dims = dims + ['frequency']

    # Fix time coordinates and add frequency
    coords = dict(signal.coords)
    new_coords = {key: values for key, values in coords.items() if values.dims != ('time',)}
    new_coords['frequency'] = f
    new_coords['time'] = t
    new_coords['s0'] = ('time', np.arange(Zxx.shape[0]))

    return xr.DataArray(Zxx, dims=dims, coords=new_coords)


def fit_spectrum_aperiodic(freqs, spec, f_range=None, **kwargs):
    """
    Fit a model of the aperiodic component of th power spectrum.
    
    Parameters
    ----------
    freqs : np.ndarray
        Frequencies at which the power spectrum is evaluated.
    spec : np.ndarray
        Power spectrum values corresponding to the frequencies.
    f_range : [fmin, fmax], optional
        Frequency range to fit the aperiodic component.
        Default is None, which uses the full range.
    kwargs : dict, optional
        Additional keyword arguments for the fitting function.

    Returns
    -------
    fit : np.ndarray
        Fitted aperiodic component of the power spectrum.
    popt : np.ndarray
        Optimized parameters of the aperiodic fit.
    """

    if f_range is None:
        f_inds = np.arange(freqs.size)
    else:
        f_inds = np.where((freqs >= f_range[0]) & 
                          (freqs <= f_range[1]))[0]

    # Fit the aperiodic component
    _, popt = fit_exp_knee(freqs[f_inds], freqs[f_inds], 
                           np.log10(spec[f_inds]), **kwargs)

    fit = exp_knee(freqs, *popt)
    fit = 10**fit  # Convert back to linear scale

    return fit, popt


def fit_spectrum_aperiodic_xr(spec, f_range=None, **kwargs):
    """
    Calculates the aperiodic component of a power spectrogram 
    stored as an xarray.DataArray, fitting along the 'frequency' dimension and broadcasting over remaining dimensions.

    Parameters
    ----------
    spec : xarray.DataArray
        Power spectrogram with a 'frequency' dimension and other dimensions to broadcast over.
    f_range : [fmin, fmax], optional
        Frequency range to fit the aperiodic component.
        Default is None, which uses the full range. 
    kwargs : dict, optional
        Additional keyword arguments for the fitting function.

    Returns
    -------
    spec_ap : xarray.DataArray
        Spectrogram with aperiodic component, with same shape as input.
    """

    # Check that 'frequency' dimension exists
    if 'frequency' not in spec.dims:
        raise ValueError("Input xarray must have a 'frequency' dimension.")

    freqs = spec['frequency'].values

    # Define a function to apply along the 'frequency' dimension
    def _fit_func(x):
        fit, _ = fit_spectrum_aperiodic(freqs, x, f_range=f_range, **kwargs)
        return fit

    # Apply along 'frequency', broadcasting over other dimensions
    spec_ap = xr.apply_ufunc(
        _fit_func,
        spec,
        input_core_dims=[['frequency']],
        output_core_dims=[['frequency']],
        vectorize=True,
        dask='parallelized'
    )

    return spec_ap
