""" Organizes data from in vitro patch clamp experiments
Created by: Drew Headley
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# for each image in imgs_xr, identify the x,y location of the center of mass. If no pixels are above 0, then return np.nan
def _center_of_mass(image: np.ndarray) -> np.ndarray:
    """Calculate the center of mass of a 2D image."""
    if np.all(image == 0):
        return np.array([np.nan, np.nan])  # No pixels above 0

    y_indices, x_indices = np.indices(image.shape)
    total_mass = np.sum(image)

    if total_mass == 0:
        return np.array([np.nan, np.nan])  # Avoid division by zero

    y_center = np.sum(y_indices * image) / total_mass
    x_center = np.sum(x_indices * image) / total_mass

    return np.array([y_center, x_center])

def format_ivic(rec: xr.DataArray) -> xr.DataArray:
    """Adds current injection information to the corresponding
    trials in the recording data.
    
    Parameters
    ----------
    rec : xr.DataArray
        The recording data with a 'trial' dimension.
        
    Returns
    -------
    rec : xr.DataArray
        The recording data with additional coordinates:
        inj_current : The current injected for each trial.
        time : The time dimension is updated to reflect the
        start of the stimulation pulse.

    """

    # get the current injection channel
    inj_ch = rec['ch_units'].to_numpy().tolist().index('pA')
    inj_current = rec.sel(channel=inj_ch)
    hold_current = inj_current.median(dim='time')
    inj_current = inj_current - hold_current  # remove DC offset

    time_data = inj_current['time'].to_numpy()
    step_start_idx = np.where(inj_current.sel(trial=0).data <=-150)[0][0]
    step_stop_idx = np.where(inj_current.sel(trial=0).data <= -150)[0][-1]
    step_start_time = time_data[step_start_idx]
    step_stop_time = time_data[step_stop_idx]
    step_dur = step_stop_time - step_start_time

    # replace time dimension with step_start_time
    rec = rec.assign_coords(time = time_data - step_start_time)

    # add inj_current to rec
    rec.coords['inj_current'] = ('trial', 
                                 inj_current.sel(time=step_start_time+0.005).data)
    rec.coords['hold_current'] = ('trial', hold_current.data)

    rec.attrs['step_dur'] = step_dur


    return rec

def format_focal_stimulation(rec: xr.DataArray, imgs: xr.DataArray) -> xr.DataArray:
    """Assigns the x,y coordinates of the center of mass of each image 
    in imgs to the corresponding trial in rec.
    
    Parameters
    ----------
    rec : xr.DataArray
        The recording data with a 'trial' dimension.
    imgs : xr.DataArray
        The images used for focal stimulation, with a 'image' 
        dimension.

    Returns
    -------
    rec : xr.DataArray
        The recording data with additional coordinates:
        'stim_x': x-coordinate for focal stimulation,
        'stim_y': y-coordinate for focal stimulation.
        'rep': The repetition number for each stimulation 
    """

    # Apply the center_of_mass function to each image
    centers = xr.apply_ufunc(
        _center_of_mass,
        imgs,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["center"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    
    # Assign coordinate names to the center dimension
    centers = centers.assign_coords(center=["y", "x"])

    # Extract the centers of mass for each pattern
    stim_cents = centers.data

    # number of trials
    n_trials = rec['trial'].size
    n_imgs = imgs['image'].size
    img_inds = np.mod(np.arange(n_trials), n_imgs)
    rep = np.ceil(np.arange(n_trials) / n_imgs).astype(int)

    # get start of first stimulation pulse
    led_pow_ch = rec['ch_names'].to_numpy().tolist().index('LEDPow')
    led_pow = rec.sel(channel=led_pow_ch, trial=0).data
    start_idx = np.where(led_pow >= 0.1)[0][0]
    start_time = rec['time'].to_numpy()[start_idx]
    stim_time = rec['time'].to_numpy() - start_time

    # replace time dimension with stim_time
    rec = rec.assign_coords(time=stim_time)

    # get the LED power level
    led_pow_lvl = np.max(led_pow)

    # add coordinate for stim_cents to iv
    rec.coords['stim_x'] = ('trial', stim_cents[img_inds,0])
    rec.coords['stim_y'] = ('trial', stim_cents[img_inds,1])
    rec.coords['rep'] = ('trial', rep)

    # add protocol attributes
    rec.attrs['led_power'] = led_pow_lvl

    return rec

def format_pattern_stim(patt_xr: xr.DataArray) -> xr.DataArray:
    """
    Adds patterened stim properties to the recording data.
    
    Parameters
    ----------
    rec : xr.DataArray
        The recording data with a 'trial' dimension.
        
    Returns
    -------
    rec : xr.DataArray
        The recording data with additional coordinates:
        ramp_peak : The peak of the light ramp.
        time : The time dimension is updated to reflect the
        start of the light ramp.

    """

    # get number of trials
    n_trials = patt_xr.sizes['trial']

    # get the LED channel
    ch_idx = patt_xr['ch_names'].values.tolist().index('LEDPow')
    led = patt_xr.sel(channel=ch_idx)
    patt_peak = led.max(dim='time').data

    time_data = led['time'].to_numpy()
    patt_start_idx = np.where(led.sel(trial=n_trials-1).data >= 0.02)[0][0]
    patt_stop_idx = np.where(led.sel(trial=0).data >= 0.02)[0][-1]
    patt_start_time = time_data[patt_start_idx]
    patt_stop_time = time_data[patt_stop_idx]
    patt_dur = patt_stop_time - patt_start_time

    # replace time dimension with step_start_time
    patt_xr = patt_xr.assign_coords(time = time_data - patt_start_time)

    # add patterned properties to patt_xr
    patt_xr.coords['patt_peak'] = ('trial', patt_peak)
    patt_xr.attrs['patt_dur'] = patt_dur

    return patt_xr

def format_var_ramp(vr_xr: xr.DataArray) -> xr.DataArray:
    """
    Adds light ramp properties to the recording data.
    
    Parameters
    ----------
    rec : xr.DataArray
        The recording data with a 'trial' dimension.
        
    Returns
    -------
    rec : xr.DataArray
        The recording data with additional coordinates:
        ramp_peak : The peak of the light ramp.
        time : The time dimension is updated to reflect the
        start of the light ramp.

    """

    # get number of trials
    n_trials = vr_xr.sizes['trial']

    # get the LED ramp channel
    ch_idx = vr_xr['ch_names'].values.tolist().index('IN 5')
    ramp_led = vr_xr.sel(channel=ch_idx)
    ramp_peak = ramp_led.max(dim='time').data

    time_data = ramp_led['time'].to_numpy()
    ramp_start_idx = np.where(ramp_led.sel(trial=n_trials-1).data >= 0.02)[0][0]
    ramp_stop_idx = np.where(ramp_led.sel(trial=0).data >= 0.02)[0][-1]
    ramp_start_time = time_data[ramp_start_idx]
    ramp_stop_time = time_data[ramp_stop_idx]
    ramp_dur = ramp_stop_time - ramp_start_time

    # replace time dimension with step_start_time
    vr_xr = vr_xr.assign_coords(time = time_data - ramp_start_time)

    # add ramp properties to vr_xr
    vr_xr.coords['ramp_peak'] = ('trial', ramp_peak)
    vr_xr.attrs['ramp_dur'] = ramp_dur


    return vr_xr


def remove_ap_xr(abf_xr: xr.DataArray,
                              voltage_channel,
                              peak_threshold: float = 0.0,
                              pre: float = 1.0,
                              post: float = 3.0,
                              interp_method: str = 'linear') -> xr.DataArray:
    """
    Remove action potentials (APs) from an ABF xarray.DataArray by replacing spike windows
    with interpolated values. Operates along the 'time' dimension and preserves other dims.

    Parameters
    ----------
    abf_xr : xr.DataArray
        ABF recording with a 'time' dimension and a 'channel' coordinate.
    voltage_channel :
        Value used to select the voltage channel (passed to abf_xr.sel(channel=...)).
    peak_threshold : float
        Voltage threshold for classifying AP samples (default 0.0).
    pre : float
        Milliseconds before spike peak to include in removal window (default 0.001).
    post : float
        Milliseconds after spike peak to include in removal window (default 0.003).
    interp_method : str
        Interpolation method for scipy.interp1d (e.g. 'linear', 'nearest', 'cubic').

    Returns
    -------
    xr.DataArray
        New DataArray with the selected channel's spikes removed (interpolated).
    """

    # precompute kernel size and convert pre/post to samples
    s_rate = abf_xr.attrs['sample_rate']
    kern_size = int(0.0005 * s_rate)
    pre = int(float(pre) / 1000)
    post = int(float(post) / 1000)

    # define the spike removal function to apply along 'time'
    def _remove_spikes_np(v):
        # voltage, time_values are 1D numpy arrays
        if v is None or v.size == 0:
            return v

        # find local maxima whose height is above peak_threshold
        peaks = find_peaks(v, height=peak_threshold)[0]
        if peaks.size == 0:
            return v

        remove_mask = np.zeros(v.shape, dtype=bool)
        for peak_idx in peaks:
            # define removal window in time around the detected peak
            start_t = t[peak_idx] - pre
            end_t = t[peak_idx] + post
            start_idx = np.searchsorted(t, start_t, side='left')
            end_idx = np.searchsorted(t, end_t, side='right') - 1
            start_idx = max(0, start_idx)
            end_idx = min(len(t) - 1, end_idx)
            remove_mask[start_idx:end_idx + 1] = True

        # if nothing left to interpolate from, return original
        if (~remove_mask).sum() < 2:
            return v

        # perform interpolation across removed samples
        keep_t = t[~remove_mask]

        v_smooth = np.convolve(v, np.ones(kern_size)/kern_size, mode='same')
        keep_v = v_smooth[~remove_mask]
        interp_fn = interp1d(keep_t, keep_v, kind=interp_method, bounds_error=False, fill_value="extrapolate")
        v_removed = v.copy()
        v_removed[remove_mask] = interp_fn(t[remove_mask])
        return v_removed

    # get time values once
    if 'time' not in abf_xr.dims and 'time' not in abf_xr.coords:
        raise ValueError("abf_xr must have a 'time' coordinate/dimension")

    t = abf_xr['time'].values

    # select the channel to process (keeps other dims if present)
    target = abf_xr.sel(channel=voltage_channel).copy()
    
    # apply function along 'time' core dim for all other dims
    cleaned = xr.apply_ufunc(
        _remove_spikes_np,
        target,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[target.dtype]
    )

    # place cleaned channel back into a copy of the original DataArray
    abf_out = abf_xr.copy()
    abf_out.loc[dict(channel=voltage_channel)] = cleaned

    return abf_out