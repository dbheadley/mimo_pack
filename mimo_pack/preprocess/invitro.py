""" Organizes data from in vitro patch clamp experiments
Created by: Drew Headley
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

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

def ivic(rec: xr.DataArray) -> xr.DataArray:
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

def focal_stimulation(rec: xr.DataArray, imgs: xr.DataArray) -> xr.DataArray:
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
