# Unit preprocessing related functions
# Author: Drew Headley
# Created: 2024-06-11
import os
import numpy as np
import pandas as pd
import xarray as xr
import dclut as dcl
import seaborn as sns
import pynapple as nap
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from tqdm import tqdm
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
from mimo_pack.plot.map import wave_map
from mimo_pack.plot.histogram import stairs_fl
from mimo_pack.plot.xarray import pcolormesh_xr
from mimo_pack.fileio.ap import load_ap_windows_raw

def plot_physpikes_summary(spks, save_dir):
    """Generate and save a summary plot of unit quality.

    This function creates a two-panel figure summarizing unit classifications and
    firing rates. The first panel is a pie chart showing the proportion of
    different unit classes (e.g., SU, MU, Noise). The second panel is a violin
    plot showing the distribution of firing rates for each class. The resulting
    figure is saved as a PDF file.

    Parameters
    ----------
    spks : pynapple.TsGroup
        The spike group object. It must contain 'class' in its metadata to
        classify units and 'rate' for firing rate analysis.
    save_dir : str
        The directory where the output PDF file will be saved. A subdirectory
        named 'qc_reports' will be created within this directory to store the
        plot.

    Raises
    ------
    ValueError
        If the 'class' column is not found in the metadata of the `spks` object.

    """
    
    # 1. Prepare Data
    # Extract class info and calculate rates
    if 'class' not in spks.metadata.columns:
        raise ValueError("The spks object does not contain 'class' metadata.")
        
    df = pd.DataFrame({
        'class': spks.get_info('class'),
        'rate': spks.rate
    })
    
    # 2. Setup Directory
    qc_dir = os.path.join(save_dir, 'qc_reports')
    os.makedirs(qc_dir, exist_ok=True)
    save_path = os.path.join(qc_dir, 'unit_quality_summary.pdf')

    # 3. Create Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Define a consistent palette (matching your notebook context if possible)
    palette = {'SU': 'tab:orange', 'MU': 'tab:green', 'Noise': 'tab:blue'}
    # Fallback for classes not in the preset palette
    unique_classes = df['class'].unique()
    for c in unique_classes:
        if c not in palette:
            palette[c] = 'gray'

    # -- Subplot 1: Pie Chart of Proportions --
    class_counts = df['class'].value_counts()
    
    # Create pie chart
    axes[0].pie(class_counts, 
                labels=[f"{c} ({class_counts[c]})" for c in class_counts.index], 
                autopct='%1.1f%%', 
                startangle=140,
                colors=[palette.get(c, 'gray') for c in class_counts.index])
    axes[0].set_title('Proportion of Unit Classes')

    # -- Subplot 2: Violin Plot of Firing Rates --
    # Transform rate to log10 scale
    df['log_rate'] = np.log10(df['rate'])

    sns.violinplot(data=df, x='class', y='log_rate', ax=axes[1], 
                   palette=palette, hue='class', legend=False,
                   cut=0)
    
    # Add strip plot on top to show individual points
    sns.stripplot(data=df, x='class', y='log_rate', ax=axes[1], 
                  color='black', alpha=0.3, size=3, jitter=True)
    
    axes[1].set_title('Firing Rate Distribution by Class')
    axes[1].set_ylabel('Firing Rate (Hz)')
    axes[1].set_xlabel('Unit Class')

    # Adjust y-ticks to reflect the original Hz values (powers of 10)
    ymin, ymax = axes[1].get_ylim()
    # Create ticks at integer intervals (representing powers of 10)
    ticks = np.arange(np.floor(ymin), np.ceil(ymax) + 1)
    axes[1].set_yticks(ticks)
    axes[1].set_yticklabels([f"$10^{{{int(t)}}}$" for t in ticks])
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.7)

    # 4. Save
    plt.tight_layout()
    fig.savefig(save_path)


def _extract_dcl_data(unit_indices, spks_dcl, dcl_path, times_sess, 
                      n_spikes=1000, window_ms=[-1.0, 2.0], exclude_channels=384):
    """
    Internal helper to extract raw spike waveforms from binary file using pre-calculated indices.
    
    Parameters
    ----------
    unit_indices : np.ndarray
        Indices of the spikes in the session.
    spks_dcl : dcl.dclut
        The DCL object for the session.
    dcl_path : str
        Path to the DCL file.
    times_sess : np.ndarray
        Time vector for the session.
    n_spikes : int, optional
        Number of spikes to sample. Default is 1000.
    window_ms : list, optional
        Time window around each spike in ms. Default is [-1.0, 2.0].
    exclude_channels : int, list, or None, optional
        Channel index or list of indices to exclude from the waveform array. 
        Useful for removing sync channels (e.g., channel 384). Default is 384.

    Returns
    -------
    waves : np.ndarray or None
        Extracted waveforms with shape (n_samples, n_channels, n_spikes).
        Returns None if no valid spikes are found.
    fs : float
        Sampling frequency estimated from times_sess.
    """
    # 1. Estimate sample rate and total samples
    fs = 1 / np.nanmedian(np.diff(times_sess))
    total_samples = times_sess.size
    
    # 2. Calculate window in samples
    pre_samples = int(abs(window_ms[0]) * fs / 1000)
    post_samples = int(window_ms[1] * fs / 1000)
    trend_samples = pre_samples + post_samples

    # 3. Filter valid spikes (Indices must allow full window extraction)
    valid_mask = (unit_indices >= pre_samples) & \
                 (unit_indices < (total_samples - post_samples))
    
    valid_indices = unit_indices[valid_mask]

    if valid_indices.size == 0:
        return None, fs

    # 4. Sample spikes
    if valid_indices.size > n_spikes:
        sample_inds = np.sort(np.random.choice(valid_indices, n_spikes, replace=False))
    else:
        sample_inds = valid_indices

    # 5. Load raw binary data
    bin_filename = spks_dcl.dcl['file']['name']
    bin_path = os.path.join(os.path.dirname(dcl_path), bin_filename)
    n_channels = spks_dcl.dcl['file']['shape'][1]
    
    try:
        # Load raw binary data: returns (Time, Channel, Window)
        waves = load_ap_windows_raw(bin_path, sample_inds.astype(np.int64), 
                                    pre_samples, post_samples, n_channels)
    except Exception as e:
        print(f"Error extracting waveforms: {e}")
        return None, fs

    # 6. Exclude specified channels (New Logic)
    if exclude_channels is not None:
        if isinstance(exclude_channels, int):
            to_remove = [exclude_channels]
        else:
            to_remove = exclude_channels
        
        # Only remove channels that actually exist in the data bounds
        current_n_channels = waves.shape[1]
        valid_remove = [c for c in to_remove if 0 <= c < current_n_channels]
        
        if valid_remove:
            waves = np.delete(waves, valid_remove, axis=1)

    # 7. Baseline Subtraction (Linear Trend)
    if waves.shape[0] > 0:
        baseline = np.linspace(waves[0,:,:], waves[-1,:,:], trend_samples)
        waves = waves - baseline
    
    return waves, fs

# Spike waveform generation
def sample_waveforms(times, bin_memmap, fs=30000, pre=1, post=2, sample_max=1000, sy_chan=384):
    """
    Load spike waveforms from binary file.
    
    Parameters
    ----------
    times : np.ndarray
        Spike times in seconds.
    bin_memmap : np.memmap
        Numpy memory mapped binary file.
    pre : numeric, optional
        Time before spike in ms. Default is 1 ms.
    post : numeric, optional
        Time after spike in ms. Default is 2 ms.
    sample_max : numeric, optional
        Maximum number of spikes to sample. Default is 1000.
    sy_chan : int, optional
        Channel number for sync signal. If None, no sync channel present.
        Default is 384.

    Returns
    -------
    waveform : np.ndarray
        Spike waveforms with shape (n_spikes, n_samples, n_channels).
    sub_flag : bool
        Flag indicating if fewer than sample_max spikes were sampled.
    """

    # convert pre and post durations to samples
    pre_samp = int(pre*fs/1000)
    post_samp = int(post*fs/1000)
    chan_num = bin_memmap.shape[1]

    # convert spike times to indices
    inds = (times*fs).astype(np.int64)
    sub_flag = True
    if inds.size > sample_max:
        inds = np.sort(np.random.choice(inds, 1000))
        sub_flag = False

    waveforms = np.zeros((inds.size, pre_samp+post_samp, chan_num))
    for i, spk in enumerate(inds):
        waveforms[i] = bin_memmap[(spk-pre_samp):(spk+post_samp), :]

    # remove sy channel if present
    if sy_chan is not None:
        waveforms = np.delete(waveforms, sy_chan, axis=2)

    return waveforms, sub_flag

def mean_waveform(times, bin_memmap, **kwargs):
    """
    Calculate the mean waveform of the spike.

    Parameters
    ----------
    times : np.ndarray
        Spike times in seconds.
    bin_memmap : np.memmap
        Numpy memory mapped binary file.
    kwargs : dict
        Keyword arguments for sample_waveforms.
    
    Returns
    -------
    mean_waveform : np.ndarray
        Mean unit waveform with shape (n_samples, n_channels)
    sub_flag : bool
        Flag indicating if fewer than sample_max spikes were sampled.
    """

    waveforms, sub_flag = sample_waveforms(times, bin_memmap, **kwargs)
    waveform = waveforms-np.mean(waveforms, axis=1)[:, np.newaxis, :]
    mean_waveform = np.mean(waveform, axis=0)
    return mean_waveform, sub_flag


def waveform_peak(waveform, scale=1):
    """
    Get the properties of the unit waveform peak
    
    Parameters
    ----------
    waveform : np.ndarray
        Unit waveform with shape (n_samples, )
    scale : numeric, optional
        Scale factor for the waveform to convert to voltage. Default is 1.
        
    Returns
    -------
    loc : int
        Location of the peak in samples
    amp : numeric
        Amplitude of the peak
    """

    loc = np.argmax(np.abs(waveform))
    amp = waveform[loc]*scale

    return loc, amp

def waveform_halfwidth(waveform, fs=30000):
    """
    Get the width of the unit waveform at half maximum
    
    Parameters
    ----------
    waveform : np.ndarray
        Unit waveform with shape (n_samples, )
    fs : numeric, optional
        Sampling frequency of the waveform in Hz. Default is 30000 Hz.
        
    Returns
    -------
    width : numeric
        Width of the waveform at half maximum in ms
    """

    up_factor = 10
    # upsample waveform by a factor of 10 with interpolation
    waveform = np.interp(np.linspace(0, waveform.size-1, waveform.size*up_factor), 
                         np.arange(waveform.size), waveform)
    loc, amp = waveform_peak(waveform)

    half_max = amp/2

    # count indices above half max starting from peak
    if amp > 0:
        left_side = np.where(waveform[loc:0:-1] < half_max)[0][0]
        right_side = np.where(waveform[loc:] < half_max)[0][0]
    else:
        left_side = np.where(waveform[loc:0:-1] > half_max)[0][0]
        right_side = np.where(waveform[loc:] > half_max)[0][0]
    
    width = (left_side + right_side - 1)/((fs*up_factor)/1000)

    return width


def classify_unit(frate, halfwidth, region='CTX'):
    """
    Classifies a unit based on waveform properties

    Parameters
    ----------
    frate : numeric
        Firing rate of the unit in Hz
    halfwidth : numeric
        Width of the waveform at half maximum in ms
    region : str, optional
        Brain region of the unit. Default is 'CTX', cortex.

    Returns
    -------
    uclass : str
        Classification of the unit. For cortex, the classes are:
        'RS' - Regular spiking
        'FS' - Fast spiking
        'UN' - Unidentified
    """

    if region == 'CTX':
        if (frate > 2) and (halfwidth < 0.15):
            uclass = 'FS'
        elif (frate < 10) and (halfwidth > 0.15):
            uclass = 'RS'
        else:
            uclass = 'UN'
    
    return uclass

def unit_occupancy(spk_t, start, end):
    """
    Calculate the occupancy of a unit in a given time period.

    Parameters
    ----------
    spk_t : np.ndarray
        Spike times in seconds.
    start : numeric
        Start time of the period in seconds.
    end : numeric
        End time of the period in seconds.

    Returns
    -------
    occupancy : float
        Occupancy of the unit in the period as a fraction of the total time.
    """
    
    total_dur = end - start # total duration of the period
    spk_dur = spk_t.max() - spk_t.min() # total duration of spiking in the period
    occupancy = spk_dur / total_dur # fraction of time the unit is occupied by spikes
    
    return float(occupancy)

def unit_refractory_violations(spk_t, ref_period=0.002, start=None, end=None):
    """
    Calculates the ratio of observed to predicted refractory period violations.
    Predicted violations are based on the mean firing rate. Can serve as a measure
    of the rate of false positive spikes.

    Parameters
    ----------
    spk_t : array
        The spike times, sorted in ascending order.
    ref_period : float, optional
        The refractory period in seconds. Default is 0.002 seconds (2 ms).
    start : float, optional
        Start time of the period in seconds. If None, uses the first spike time.
    end : float, optional
        End time of the period in seconds. If None, uses the last spike time.

    Returns
    -------
    r_fp : float
        The ratio of observed to predicted refractory period violations
    """
    
    if start is None:
        start = spk_t[0]
    if end is None:
        end_t = spk_t[-1]

    # ensure spike times are within the specified period
    spk_t = spk_t[(spk_t >= start) & (spk_t <= end)]
    if spk_t.size == 0:
        return np.nan

    num_spks = spk_t.size
    dur = end-start # total duration of spiking
    viol_count = np.sum(np.diff(spk_t)<=ref_period) # number of refractory period violations
    refract_time = 2*ref_period*num_spks # total potential time for refractory period violations
    spk_rate = num_spks/dur # mean firing rate, irrespective of refractory period
    viol_rate = viol_count/refract_time # firing rate just during the refractory period
    r_fp = viol_rate/spk_rate # ratio of observed to predicted violations
    return float(r_fp)


# based the algorithm from the Allen Institute spike quality metrics code
# which is based on a measure proposed in Hill et al. 2011 J Neurosci 31: 8699-8705
# I have modified it to account for the lost spikes when calculating the 
# total spike count
def amp_cutoff(spk_a):
    """
    Calculates the amplitude cutoff for a given unit

    Parameters
    ----------
    spk_a : array
        The spike amplitudes for a unit

    Returns
    -------
    miss_prob : float
        The probability that a spike will be missed due to amplitude cutoff
    """
    dist, bins = np.histogram(spk_a, bins=50)
    dist = gaussian_filter1d(dist, 3) # smooth the distribution
    peak_idx = np.argmax(dist) # find the peak

    # find the first point in the distribution above the peak that
    # falls below the probability density for the lowest amplitude.
    # If the true distribution of spike amplitudes is symmetric, then
    # then the area under from this point to the maximum amplitude should
    # be equal to the 
    g = np.argmin(np.abs(dist[peak_idx:]-dist[0]))+peak_idx

    # calculate the area under the curve from the end of the distribution
    miss_count = np.sum(dist[g:]) # calculate area under the curve from the end of the distribution

    total_count = miss_count+spk_a.size # to estimate total count, add the missed to the observed
    miss_prob = miss_count/total_count # get proportion of total spikes missed

    # have max_prob cutoff at 0.5
    miss_prob = np.min([miss_prob, 0.5])

    return float(miss_prob)

def waveform_spread(peak_map, x_pos, y_pos):
    """
    Calculates the spread of the waveform across channels
    by measuring the radius of the halfwidth.

    Parameters
    ----------
    peak_map : np.ndarray
        The peak values at each of the channel locations for the waveform.
    x_pos : np.ndarray
        The x coordinates of the channels.
    y_pos : np.ndarray
        The y coordinates of the channels.
    Returns
    -------
    spread : float
        The spread of the waveform across channels in microns.
    """

    peak_map_abs = np.abs(peak_map)  # get absolute values of peak map

    # get the peak channel
    peak_chan = np.argmax(peak_map_abs)
    peak_x = x_pos[peak_chan]
    peak_y = y_pos[peak_chan]

    # get the furthest channel whose peak value is above half the maximum
    half_max = np.max(peak_map_abs)/2
    above_half = np.where(peak_map_abs > half_max)[0]
    if above_half.size == 0:
        return 0.0  # no channels above half max, spread is zero
    
    chan_dists = np.sqrt((x_pos[above_half] - peak_x)**2 + 
                         (y_pos[above_half] - peak_y)**2)
    spread = np.max(chan_dists)  # maximum distance from peak channel to any channel above half max
    return float(spread)

def waveform_snr(waveforms, mode='resid', mean_waveform=None):
    """
    Calculate the signal-to-noise ratio (SNR) of a spike waveform.
    
    Parameters
    ----------
    waveforms : np.ndarray, (n_samples, n_channels, n_spikes)
        The spike waveforms.
    mode : str, optional
        The type of SNR to calculate. Options:
        'resid' - Residual SNR, ratio of the standard deviation of the 
        mean waveform to the standard deviation of the residuals. The 
        channel with the largest waveform is used. Inspired by 
        Joshua et al. 2007. Default is 'resid'.
    mean_waveform : np.ndarray, optional
        The mean waveform to use for calculating the residuals. If None, 
        it will be calculated from the waveforms. Default is None.

    Returns
    -------
    snr : float
        The signal-to-noise ratio of the spike waveform.
    """
    
    if waveforms.ndim != 3:
        raise ValueError('Waveforms must be a 3D array with shape ' +
                         '(n_samples, n_channels, n_spikes)')
    if mode == 'resid':
        # calculate the residuals of the waveforms
        if mean_waveform is None:
            mean_waveform = np.mean(waveforms, axis=2)

        # get channel with the largest waveform
        peak_chan = np.argmax(np.max(np.abs(mean_waveform), axis=0))
        peak_waveform = mean_waveform[:, peak_chan]

        # calculate the residuals
        residuals = waveforms[:, peak_chan, :] - peak_waveform[:, np.newaxis]

        # calculate the SNR
        mean_std = np.std(peak_waveform)
        resid_std = np.std(residuals)
        snr = mean_std / resid_std

    else:
        raise ValueError("Unsupported SNR type: {}".format(mode))
    
    return float(snr)

def waveform_envelope(waveform):
    """
    Calculate the envelope of a waveform using the Hilbert transform.

    Parameters
    ----------
    waveform : np.ndarray or xr.DataArray
        The waveform to calculate the envelope for. Has shape (n_samples, n_channels).

    Returns
    -------
    envelope : np.ndarray or xr.DataArray
        The envelope of the waveform.
    """

    if isinstance(waveform, xr.DataArray):
        dim = 'time' if 'time' in waveform.dims else waveform.dims[0]
        axis = waveform.get_axis_num(dim)
        analytic_signal = hilbert(waveform.values, axis=axis)
        envelope = np.abs(analytic_signal)
        return waveform.copy(data=envelope)

    analytic_signal = hilbert(waveform, axis=0)
    envelope = np.abs(analytic_signal)

    return envelope

def add_waveform_metrics(spks, dcl_file, n_spikes=1000, wave_period=[-1.0, 2.0], 
                         snr_mode='resid', exclude_channels=384, verbose=True):
    """
    Add waveform metrics to each unit in the spike sorting output.

    Parameters
    ----------
    spks : Tsd
        Spike sorting output where keys are unit IDs and values are 
        pynapple Ts or Tsd objects containing spike times.
    dcl_file : str
        Path to the DCL file associated with the spike sorting output.
    n_spikes : int, optional
        Number of spikes to sample for waveform extraction. Default is 1000.
    wave_period : list, optional
        Time window around each spike to extract waveform in ms. 
        Default is [-1.0, 2.0].
    snr_mode : str, optional
        Method for calculating SNR. Default is 'resid'.
    exclude_channels : int, list, or None, optional
        Channel index or list of indices to exclude from the waveform array. 
        Useful for removing sync channels (e.g., channel 384). Default is 384.
    verbose : bool, optional
        Whether to display progress information. Default is True.  

    Returns
    -------
    spks : Tsd
        Updated spike sorting output with waveform metrics added to each unit's info.
    """

    if verbose: print("Extracting waveforms and computing properties...")

    if not os.path.isfile(dcl_file):
        raise FileNotFoundError(f"DCL file not found: {dcl_file}")

    spks_dcl = dcl.dclut(dcl_file)
    times_sess = spks_dcl.scale_values('time')
    x_pos = spks_dcl.scale_values('ch_x')
    y_pos = spks_dcl.scale_values('ch_y')
    shank = spks_dcl.scale_values('ch_shank')
    
    # Handle channel exclusion for coordinate arrays to match waveform data
    if exclude_channels is not None:
        if isinstance(exclude_channels, int):
            ex_list = [exclude_channels]
        else:
            ex_list = exclude_channels
        
        # Create mask of channels to keep
        n_total_ch = len(x_pos)
        keep_mask = np.ones(n_total_ch, dtype=bool)
        valid_remove = [c for c in ex_list if 0 <= c < n_total_ch]
        keep_mask[valid_remove] = False
        
        # Apply mask to coordinates
        x_pos = x_pos[keep_mask]
        y_pos = y_pos[keep_mask]
        shank = shank[keep_mask]

    sort_inds = np.lexsort((x_pos, y_pos, shank))
    x_pos = x_pos[sort_inds]
    y_pos = y_pos[sort_inds]
    shank = shank[sort_inds]

    # Storage for metadata
    wave_list = []
    snr_list = []
    spread_list = []
    cutoff_list = []
    amp_hist_list = []

    iter_list = tqdm(spks.keys(), desc="Processing Units") if verbose else spks.keys()

    for uid in iter_list:
        spks_dcl.reset()
        unit = spks[uid]
        
        # Check if we have indices stored in the Tsd object
        if isinstance(unit, nap.Tsd):
            unit_indices = unit.values
        else:
            # Fallback for generic Ts objects
            unit_times = unit.times()
            unit_indices = np.searchsorted(times_sess, unit_times)

        # 1. Extract Raw Data (Passing exclude_channels)
        waves, fs = _extract_dcl_data(unit_indices, spks_dcl, dcl_file, times_sess, 
                                      n_spikes=n_spikes, window_ms=wave_period,
                                      exclude_channels=exclude_channels)
        
        if waves is None:
            wave_list.append(None)
            snr_list.append(np.nan)
            spread_list.append(np.nan)
            cutoff_list.append(np.nan)
            amp_hist_list.append(None)
            continue

        waves = waves[:, sort_inds, :]  # Sort channels to match coordinate arrays
        # 2. Compute Mean and Spatial Peak
        mean_wave = np.mean(waves, axis=2) # (Samples, Channels)
        amp_map_val = np.linalg.norm(mean_wave, axis=0)
        peak_chan = np.argmax(amp_map_val)
        
        # Identify 8 Nearest Neighbors (plus peak itself)
        dists = (x_pos - x_pos[peak_chan])**2 + (y_pos - y_pos[peak_chan])**2
        neighbor_inds = np.argsort(dists)[:8]
        
        # Sort neighbors by amplitude
        local_amps = np.linalg.norm(mean_wave[:, neighbor_inds], axis=0)
        sorted_local_inds = neighbor_inds[np.argsort(local_amps)]
        
        # Data for QC metrics
        peak_wave = mean_wave[:, sorted_local_inds]
        raw_peak_waves = waves[:, sorted_local_inds, :] # (Samples, 8_chans, Spikes)
        
        # 3. Compute Metrics
        # A. SNR
        snr = waveform_snr(raw_peak_waves, mode=snr_mode, mean_waveform=peak_wave)
        snr_list.append(snr)
        
        # B. Spread
        spread = waveform_spread(amp_map_val, x_pos, y_pos)
        spread_list.append(spread)
        
        # C. Amplitude Cutoff (Projection Method)
        template = peak_wave.flatten()
        template_norm = np.linalg.norm(template)
        unit_template = template / template_norm if template_norm > 0 else template

        # Flatten raw spikes: (Samples, Channels, Spikes) -> (Spikes, Samples*Channels)
        spikes_flat = raw_peak_waves.transpose(2, 0, 1).reshape(raw_peak_waves.shape[2], -1)
        
        # Project spikes onto the template direction
        wave_amps = np.dot(spikes_flat, unit_template)

        cutoff = amp_cutoff(wave_amps)
        cutoff_list.append(cutoff)
        
        # Store Amp Histogram data
        ahist, abins = np.histogram(wave_amps, bins=50, density=True)
        amp_hist_list.append({'hist': ahist, 'bins': abins})

        # 4. Construct Metadata Object
        full_xr = xr.DataArray(
            mean_wave,
            dims=('time', 'channel'),
            coords={
                'time': np.arange(mean_wave.shape[0]) - int(-wave_period[0] * fs/1000),
                'channel': np.arange(mean_wave.shape[1]),
                'ch_x': ('channel', x_pos),
                'ch_y': ('channel', y_pos),
                'ch_shank': ('channel', shank)
            }
        )

        wave_list.append({
            'waveform': peak_wave,      # Small dense mean (Samples, 8)
            'inds': sorted_local_inds,  # Indices of the 8 channels
            'full_waveform': full_xr,   # Full probe xarray
            'x': x_pos[peak_chan],
            'y': y_pos[peak_chan],
            'shank': shank[peak_chan]
        })

    spks.set_info(waveform=wave_list)
    spks.set_info(WaveformSNR=snr_list, 
                  WaveformSpread=spread_list, 
                  AmplitudeCutoff=cutoff_list,
                  AmplitudeHist=amp_hist_list)
    
    valid_waves = [w for w in wave_list if w is not None]
    if valid_waves:
        spks.set_info(x=[w['x'] if w else np.nan for w in wave_list])
        spks.set_info(y=[w['y'] if w else np.nan for w in wave_list])
        spks.set_info(shank=[w['shank'] if w else np.nan for w in wave_list])

    return spks


def run_qc_report(spks, report_path=None, ref_period=0.002, verbose=True):
    """
    Calculates time-based QC metrics and generates PDF reports.

    Parameters
    ----------
    spks : pynapple.TsGroup
        The spike group object containing unit spike times.
    report_path : str, optional
        Directory path to save the PDF reports. If None, no reports are generated.
    ref_period : float, optional
        Refractory period in seconds for violation calculation. Default is 0.002.
    verbose : bool, optional
        Whether to display progress bar. Default is True.

    Returns
    -------
    spks : pynapple.TsGroup
        The input object updated with 'Occupancy' and 'RefractoryViolations' metadata.
    """

    if verbose: print("Calculating time metrics and generating reports...")
    
    if report_path and not os.path.exists(report_path):
        os.makedirs(report_path)

    occupy = []
    ref_fp = []
    
    has_waves = 'WaveformSNR' in spks.metadata_columns
    
    iter_list = tqdm(spks.keys(), desc="QC Analysis") if verbose else spks.keys()

    for uid in iter_list:
        unit = spks[uid]
        times = unit.times()
        start_sess = unit.time_support['start'][0]
        end_sess = unit.time_support['end'][0]

        # 1. Time-Based Metrics
        occ = unit_occupancy(times, start=start_sess, end=end_sess)
        ref = unit_refractory_violations(times, ref_period=ref_period, start=start_sess, end=end_sess)
        
        occupy.append(occ)
        ref_fp.append(ref)

        # 2. Generate Report
        if report_path and has_waves:
            info = spks.get_info(['waveform', 'class', 'AmplitudeHist', 'AmplitudeCutoff', 'WaveformSpread', 'WaveformSNR']).loc[uid]
            wf_data = info['waveform']
            if wf_data is None: continue

            fig = plt.figure(figsize=(8.5, 11))
            gs = GridSpec(6, 6, figure=fig)
            
            # Table
            ax_tbl = fig.add_subplot(gs[0:3, 0:2])
            ax_tbl.axis('off')
            u_class = info['class'] if 'class' in info else 'Unknown'
            row_labels = ['Unit ID', 'Class', 'Occupancy', 'Refractory violations',
                          'Amplitude cutoff', 'Waveform spread', 'Waveform SNR']
            cell_text = [[f'{uid}'], [u_class], [f'{occ:.2f}'], 
                         [f'{ref:.3f}'], [f"{info['AmplitudeCutoff']:.3f}"], 
                         [f"{info['WaveformSpread']:.1f}"], [f"{info['WaveformSNR']:.2f}"]]
            ax_tbl.table(cellText=cell_text, rowLabels=row_labels, loc='center')

            # Waveforms
            ax_waves = fig.add_subplot(gs[0:3, 2:4])
            full_xr = wf_data['full_waveform']
            x_near = full_xr.ch_x.values[wf_data['inds']]
            y_near = full_xr.ch_y.values[wf_data['inds']]
            wave_map(wf_data['waveform'], x_pos=x_near, y_pos=y_near,
                     ax=ax_waves, x_scale=0.3, y_scale=0.5)
            ax_waves.set_title('Example waveforms')

            # Spread (Composite)
            ax_spread = fig.add_subplot(gs[0:3, 4:6])
            peak_x = wf_data['x']
            peak_shank = wf_data['shank']
            shank_xr = full_xr.where(full_xr.ch_x == peak_x, drop=True)
            shank_xr = shank_xr.sortby('ch_y')
            shank_xr.data = np.sqrt(np.abs(shank_xr.data))*np.sign(shank_xr.data)
            max_val = np.max(np.abs(shank_xr.data))
            pcolormesh_xr(shank_xr, row_coord='ch_y', col_coord='time', cmap='bwr', 
                          vmin=-max_val, vmax=max_val, ax=ax_spread)
            ax_spread.set_title(f'Waveform Composite shank {peak_shank}')

            # Firing Rate Hist
            ax_fr = fig.add_subplot(gs[3, 0:6])
            fr_hist, bins = np.histogram(times, bins=np.arange(start_sess, end_sess, 60))
            stairs_fl(fr_hist/60, bins, fill_color='black', edge_color='black', baseline=0, ax=ax_fr)
            ax_fr.set_title('Firing rate histogram')
            ax_fr.set_xlim(start_sess, end_sess)

            # ISI Hist
            ax_isi = fig.add_subplot(gs[4:6, 0:3])
            isi = np.diff(times)
            isi_hist, isi_bin = np.histogram(isi, bins=10**np.arange(-4, 1, 0.1), density=True)
            stairs_fl(isi_hist, isi_bin, fill_color='black', edge_color='black', baseline=0, ax=ax_isi)
            ax_isi.set_xscale('log')
            ax_isi.set_title('ISI histogram')

            # Amp Hist
            ax_amp = fig.add_subplot(gs[4:6, 3:6])
            ahist_data = info['AmplitudeHist']
            stairs_fl(ahist_data['hist'], ahist_data['bins'], fill_color='black', 
                      edge_color='black', baseline=0, ax=ax_amp)
            ax_amp.set_title('Amplitude histogram')

            fig.tight_layout()
            fig.savefig(os.path.join(report_path, f'unit_{uid}_qc_report.pdf'), bbox_inches='tight')
            plt.close(fig)

    spks.set_info(Occupancy=occupy, RefractoryViolations=ref_fp)
    return spks