# Functions for spectral analysis
# Author: Drew Headley
# Date: 2025-06-16

import warnings
import pywt
import xarray as xr
import numpy as np
import pandas as pd
import scipy.signal as ss
from mimo_pack.math.curvefitting import fit_exp_knee, exp_knee

def stft_xr(signal, window=1, **kwargs):
    """
    Apply Short-Time Fourier Transform (STFT) to an xarray signal 
    and return an xarray with a frequency dimension.

    Parameters
    ----------
    signal : xarray.DataArray
        Must have a time dimension. STFT will be applied along 
        this dimension.
    window : float
        Duration (in seconds) of each STFT window. Default is 1 second.
    kwargs : dict, optional
        Additional arguments to pass to scipy.signal.stft 
        (e.g., nperseg, noverlap, etc.).

    Returns
    -------
    xarray.DataArray
        STFT result with same non-time dimensions and an added 
        'frequency' dimension.
    """

    if 'time' not in signal.dims:
        raise ValueError("Input xarray must have a 'time' dimension.")
    
    if not isinstance(window, np.ndarray):
        window = np.array([window])

    # Extract time and sampling frequency
    time = signal['time'].values
    dt = np.nanmedian(np.diff(time))
    fs = 1.0 / dt

    arr = signal.to_numpy()
    dims = signal.dims
    time_axis = np.where(np.array(dims) == 'time')[0][0]

    hop = int(window * fs)
    mfft = kwargs.get('mfft', int(window * fs))
    kwargs['mfft'] = mfft

    win_func = ss.windows.gaussian(mfft, std=mfft/5.0, sym=True)
    stft = ss.ShortTimeFFT(win_func, hop, fs, **kwargs)
    
    f = stft.f
    t = stft.t(arr.shape[time_axis])

    Zxx = stft.spectrogram(arr, axis=time_axis)
        
    # rearrange spectrogram array to have
    # original dimenions and with frequency last
    Zxx = Zxx.swapaxes(-1,time_axis)

    # Prepare output dims and coords
    dims = list(signal.dims)
    dims = dims + ['frequency']

    # Fix time coordinates and add frequency
    coords = dict(signal.coords)
    new_coords = {key: values for key, values in coords.items() 
                  if values.dims != ('time',)}
    new_coords['frequency'] = f
    new_coords['time'] = t
    new_coords['s0'] = ('time', np.arange(Zxx.shape[time_axis]))

    return xr.DataArray(Zxx, dims=dims, coords=new_coords)


def wavelet_xr(signal, freqs=np.power(2,np.arange(0,7,0.25)), wavelet='cmor2.5-1.0',
               remove_dc=True, verbose=True):
    """
    Apply a wavelet transform an xarray signal

    Parameters
    ----------
    signal : xarray.DataArray
        Must have a 'time' dimension.
    freqs : np.ndarray, optional
        Frequencies to use for the wavelet transform. 
        Default is from 1 to 128 Hz on a logarithmic scale.
    wavelet : str, optional
        Wavelet type to use for the transform.
        Default is 'cmor2.5-1.0' (complex Morlet wavelet).
    remove_dc : bool, optional
        Whether to remove the DC offset from the signal before
        applying the wavelet transform. Default is True.
    verbose : bool, optional
        Whether to print warnings about the size of the resulting
        wavelet transform. Default is True.

    Returns
    -------
    xarray.DataArray
        wavelet result with same non-time dimensions and an 
        added 'frequency' dimension appended.
    """

    if 'time' not in signal.dims:
        raise ValueError("Input xarray must have a 'time' dimension.")
    
    if not isinstance(freqs, np.ndarray):
        freqs = np.array(freqs)

    # calculate size of the resulting array from the wavelet transform
    # an give a warning if the size exceeds 20 GB
    est_size = (signal.size * freqs.size * 
                np.dtype(np.complex64).itemsize) / 1e9  # in GB
    if verbose and est_size > 20:
        warnings.warn(f"The resulting wavelet transform will be {est_size} GB.", 
                      UserWarning)

    # Extract time and sampling frequency
    time = signal['time'].values
    dt = np.nanmedian(np.diff(time))
    fs = 1.0 / dt

    arr = signal.to_numpy()
    dims = signal.dims

    time_axis = np.where(np.array(dims) == 'time')[0][0]

    # Create a continuous wavelet transform object
    # We will use the complex Morlet wavelet
    w = pywt.ContinuousWavelet(wavelet)

    # convert frequencies to scales
    scales = pywt.frequency2scale(w, freqs) * fs 

    # remove DC offset to minimize edge effects
    if remove_dc:
        arr = arr - np.mean(arr, axis=time_axis, keepdims=True) 

    # Continuous wavelet transform
    Wxx, _ = pywt.cwt(arr, scales, w, sampling_period=dt, 
                      axis=time_axis, method='fft')
        
    # rearrange spectrogram array to have original organization,
    # and with frequency last
    Wxx = Wxx.transpose(np.roll(np.arange(Wxx.ndim), -1))

    # Prepare output dims and coords
    dims = list(signal.dims)
    dims = dims + ['frequency']

    # Fix time coordinates and add frequency
    new_coords = dict(signal.coords)
    new_coords['frequency'] = freqs

    return xr.DataArray(Wxx, dims=dims, coords=new_coords)


def wavelet_phamp_xr(wavelet):
    """
    Calculate the phase and amplitude of a wavelet transform stored as an xarray.DataArray.

    Parameters
    ----------
    wavelet : xarray.DataArray
        Wavelet transform with a 'frequency' dimension and other dimensions to broadcast over.

    Returns
    -------
    amplitude_xr : xarray.DataArray
        Amplitude of the wavelet transform, with same shape as input.
    phase_xr : xarray.DataArray
        Phase of the wavelet transform, with same shape as input.
    """

    # Check that 'frequency' dimension exists
    if 'frequency' not in wavelet.dims:
        raise ValueError("Input xarray must have a 'frequency' dimension.")

    # Calculate phase and amplitude
    wave = wavelet.to_numpy()
    phase = np.angle(wave)
    amplitude = np.abs(wave)

    # Create DataArrays for phase and amplitude
    phase_xr = xr.DataArray(phase, dims=wavelet.dims, coords=wavelet.coords)
    amplitude_xr = xr.DataArray(amplitude, dims=wavelet.dims, coords=wavelet.coords)

    return amplitude_xr, phase_xr


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

def compute_spike_phamp_hist(spks, lfp, bins=(4, 9), freqs=[52]):
    """
    Compute 2D histograms of spike phase and amplitude percentiles for 
    each unit grouped by LFP channel, for one or more frequencies.

    Parameters
    ----------
    spks : pynapple.TsGroup
        Spike units object with 'lfp_channel' metadata field.
    lfp : xarray.DataArray
        LFP data with channel dimension.
    bins : tuple of int
        Number of bins for amplitude percentile and phase, e.g. (4, 9).
    freqs : float, int, or list/array
        Frequency or frequencies (Hz) to extract from wavelet. Default is [52].

    Returns
    -------
    pd.Series
        Series of xarray.DataArray histograms indexed by unit, 
        each with dims ('amplitude', 'phase', 'frequency').
    """
    if not isinstance(freqs, (list, np.ndarray)):
        freqs = [freqs]
    freqs = np.array(freqs)

    spks_grp = spks.getby_category('lfp_channel')
    hist_grp = []
    for ch in spks_grp:
        spk_sel = spks_grp[ch]
        lfp_ch = lfp.sel(channel=ch)
        lfp_wave = wavelet_xr(lfp_ch, freqs=freqs)
        t_dim = np.where(np.array(lfp_wave.dims) == 'time')[0][0]
        f_dim = np.where(np.array(lfp_wave.dims) == 'frequency')[0][0]
        wave_amp, wave_ph = wavelet_phamp_xr(lfp_wave)

        # convert wave_amp to percentiles along time for each frequency
        wave_amp_pr = wave_amp.copy(deep=True)
        # argsort twice to get ranks, then normalize to percentiles
        amp_np = wave_amp.to_numpy()
        amp_rank = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), t_dim, amp_np)
        amp_pr = amp_rank / (amp_np.shape[t_dim] - 1 + 1e-12)
        wave_amp_pr.data = amp_pr

        hist = []
        for spk_i in spk_sel:
            times_i = spk_sel[spk_i].times()
            # Select nearest time points for all frequencies
            amp_pr_sel = wave_amp_pr.sel(time=times_i, method='nearest').to_numpy()
            ph_sel = wave_ph.sel(time=times_i, method='nearest').to_numpy()
            # amp_pr_sel, ph_sel shape: (n_times, n_freqs) or (n_freqs, n_times) depending on dims
            # Ensure shape is (n_times, n_freqs)
            if amp_pr_sel.shape[0] != len(times_i):
                amp_pr_sel = amp_pr_sel.T
                ph_sel = ph_sel.T

            # Remove spikes with amplitude percentile > 0.999 for each frequency
            hists = []
            for fi in range(len(freqs)):
                amp_f = amp_pr_sel[:, fi]
                ph_f = ph_sel[:, fi]
                mask = amp_f <= 0.999
                amp_f = amp_f[mask]
                ph_f = ph_f[mask]
                h2d, amp_edges, ph_edges = np.histogram2d(
                    amp_f, ph_f, bins=bins, range=[[0, 1], [-np.pi, np.pi]]
                )
                hists.append(h2d)

            # Stack along frequency dimension
            hists = np.stack(hists, axis=-1)
            # Create xarray for this unit: dims (amplitude, phase, frequency)
            hist_xr = xr.DataArray(
                hists,
                dims=('amplitude', 'phase', 'frequency'),
                coords={
                    'amplitude': 0.5 * (amp_edges[:-1] + amp_edges[1:]),
                    'phase': 0.5 * (ph_edges[:-1] + ph_edges[1:]),
                    'frequency': freqs
                },
                name='count'
            )
            hist.append(hist_xr)

        hist_grp.append(pd.Series(hist, index=spk_sel.index, name='entrainment_hist'))

    hist_full = pd.concat(hist_grp)
    hist_full = hist_full.sort_index()
    return hist_full


def ppc2(phases):
    """
    Alternative calculation of the Pairwise Phase Consistency (PPC2)
    for a set of phase values.

    Created by Greg Glickert

    Parameters
    ----------
    phases: A 1D numpy array 
        Phase values in radians. The shape should
        be (n_observations,). For example, these could be the phases
        of spike times relative to an LFP oscillation.

    Returns
    -------
    ppc2: float
        The PPC value, a float. This is an unbiased estimator of the squared
        Phase-Locking Value (PLV^2) as described in Vinck et al. (2010).
        Returns 0 if there are fewer than two phase values.
    """
    # Calculate PPC2 according to Vinck et al. (2010), Equation 6
    n = len(phases)

    if n <= 1:
        print("Warning: Cannot calculate PPC with less than 2 observations.")
        return np.nan


    # Convert phases to unit vectors in the complex plane
    unit_vectors = np.exp(1j * phases)

    # Calculate the resultant vector
    resultant_vector = np.sum(unit_vectors)

    # PPC2 = (|∑(e^(i*φ_j))|² - n) / (n * (n - 1))
    ppc2 = (np.abs(resultant_vector) ** 2 - n) / (n * (n - 1))

    return ppc2