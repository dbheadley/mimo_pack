# Analysis of behavioral states
# Authors: Drew B Headley
# Date: 2025-10-07

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.mixture import GaussianMixture
from mimo_pack.analysis.spectral import stft_xr, fit_spectrum_aperiodic_xr


def _find_gmm_crossover(data, n_components=2):
    """
    Fit a 2-component Gaussian Mixture Model and find the crossover point.

    Parameters
    ----------
    data : np.ndarray
        1D array of values to fit.
    n_components : int, optional
        Number of Gaussian components (default is 2).

    Returns
    -------
    crossover : float
        The value where the two Gaussian densities are equal.
    """
    # Remove NaNs and reshape for sklearn
    valid_data = data[~np.isnan(data)].reshape(-1, 1)
    
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(valid_data)
    
    # Get means and stds
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()
    
    # Sort by mean
    order = np.argsort(means)
    mu1, mu2 = means[order]
    std1, std2 = stds[order]
    w1, w2 = weights[order]
    
    # Find crossover by solving for where weighted Gaussians are equal
    # Search between the two means
    x_range = np.linspace(mu1, mu2, 1000)
    
    # Weighted Gaussian densities
    g1 = w1 * np.exp(-0.5 * ((x_range - mu1) / std1) ** 2) / std1
    g2 = w2 * np.exp(-0.5 * ((x_range - mu2) / std2) ** 2) / std2
    
    # Find crossover (where g1 and g2 are closest)
    diff = np.abs(g1 - g2)
    crossover_idx = np.argmin(diff)
    crossover = x_range[crossover_idx]
    
    return crossover


def states_lfp_only(lfp, delta=[1, 4], theta=(6,9), high=(20,80), noise=(150, 200),
                    exclude_rem=True):
    """
    Simple state scoring based on LFP band power.
    Uses z-scored band power in delta, theta, and high gamma bands to classify
    time points into active, wake, nrem, and rem states.
    Parameters
    ----------
    lfp : xarray.DataArray
        LFP signal with dimensions (time,).
    delta : list, optional
        Frequency range for delta band (default is [1, 4] Hz).
    theta : tuple, optional
        Frequency range for theta band (default is (6, 9) Hz).
    high : tuple, optional
        Frequency range for high gamma band (default is (20, 80) Hz).
    noise : tuple, optional
        Frequency range for noise band (default is (150, 200) Hz).
    exclude_rem : bool, optional
        If True, identifies REM epochs (default is True).
    Returns
    -------
    states : xarray.DataArray
        States array with dimensions (time, state) and state names in states.state.
    spec_z : np.ndarray
        Matrix of shape (time, band) containing z-score band power values.
    """

    spec = stft_xr(lfp, window=5)
    spec = spec.rolling(time=10, min_periods=1, center=True).mean()
    
    aper = fit_spectrum_aperiodic_xr(spec, f_range=(0, 80))
    aper = aper.ffill(dim='time').bfill(dim='time')
    per = (spec-aper)/aper
    
    delta_pow = np.log(aper.sel(frequency=slice(delta[0], delta[1])).mean('frequency'))
    theta_pow = per.sel(frequency=slice(theta[0], theta[1])).mean('frequency')
    high_pow = per.sel(frequency=slice(high[0], high[1])).mean('frequency')
    noise_pow = aper.sel(frequency=slice(noise[0], noise[1])).mean('frequency')

    # Find GMM crossover thresholds for delta and noise power
    delta_thresh = _find_gmm_crossover(delta_pow.values)
    noise_thresh = _find_gmm_crossover(noise_pow.values)

    if exclude_rem:
        n_states = 3
        state_names = ['wake', 'active', 'nrem']
    else:
        n_states = 4
        state_names = ['wake', 'active', 'nrem', 'rem']

    spec_z = zscore(np.vstack([delta_pow, theta_pow, high_pow, noise_pow]).T, axis=0)

    # active: high noise (above threshold)
    # nrem: high delta (above threshold), low noise (below threshold)
    # rem: high theta, low delta (below threshold), low noise (below threshold)
    # wake: any remaining time points

    states = np.zeros(spec_z.shape[0], dtype=int) # default is wake

    states[noise_pow.values > noise_thresh] = 1 # active
    states[(delta_pow.values > delta_thresh) 
           & (noise_pow.values <= noise_thresh)] = 2  # nrem
    if not exclude_rem:
        states[(theta_pow.values > (theta_pow.values.max()/4)) 
               & (delta_pow.values < delta_thresh) 
               & (noise_pow.values < noise_thresh)] = 3  # rem

    state_mat = np.zeros((spec_z.shape[0], n_states), dtype=int)
    state_mat[np.arange(spec_z.shape[0]), states] = 1

    # create xarray of states
    states = xr.DataArray(state_mat, dims=['time', 'state'], 
                          coords={'time': spec.time, 'state': state_names})

    return states, spec_z

def states_lfp_mot(lfp, mot, delta=[1, 4], theta=(6,9), high=(20,80), exclude_rem=False):
    """
    Simple state scoring based on LFP band power.
    Uses z-scored band power in delta, theta, and high gamma bands to classify
    time points into wake, nrem, and rem states.

    Parameters
    ----------
    lfp : xarray.DataArray
        LFP signal with dimensions (time,).
    mot : xarray.DataArray
        Motion signal with dimensions (time,). Only motion data should be present
        in this array.
    delta : list, optional
        Frequency range for delta band (default is [1, 4] Hz).
    theta : tuple, optional
        Frequency range for theta band (default is (6, 9) Hz).
    high : tuple, optional
        Frequency range for high gamma band (default is (20, 80) Hz).
    exclude_rem : bool, optional
        If True, only classify into wake and nrem states (default is False).
    Returns
    -------
    states : xarray.DataArray
        States array with dimensions (time, state) and state names in states.state.
    state_z : np.ndarray
        Matrix of shape (time, band + 1) containing z-score band power values and motion signal.
    """

    spec = stft_xr(lfp, window=5)
    spec = spec.rolling(time=10, min_periods=1, center=True).mean()
    
    aper = fit_spectrum_aperiodic_xr(spec, f_range=(0, 80))
    aper = aper.ffill(dim='time').bfill(dim='time')
    per = (spec-aper)/aper
    
    # process movement signal
    mot_rate = 1 / np.nanmedian(np.diff(mot.time.values))
    mot = mot.rolling(time=int(mot_rate*20), min_periods=1, center=True).mean()

    # sample motion at spectral time points
    mot_interp = np.interp(spec.time.values, mot.time.values, mot.values.flatten())

    delta_pow = np.log(aper.sel(frequency=slice(delta[0], delta[1])).mean('frequency'))
    theta_pow = per.sel(frequency=slice(theta[0], theta[1])).mean('frequency')
    high_pow = per.sel(frequency=slice(high[0], high[1])).mean('frequency')

    if exclude_rem:
        n_states = 3
        state_names = ['active', 'wake', 'nrem']
    else:
        n_states = 4
        state_names = ['active', 'wake', 'nrem', 'rem']

    spec_z = zscore(np.vstack([delta_pow, theta_pow, high_pow, mot_interp]).T, 
                    axis=0, nan_policy='omit')

    # active: high motion
    # wake: high gamma, low delta
    # nrem: high delta, low gamma
    # rem: high theta, low delta
    states = np.ones(spec_z.shape[0], dtype=int)

    states[(spec_z[:,0] > 0) & (spec_z[:,2] < 0) & (spec_z[:,3] < 0)] = 2  # nrem
    #states[(spec_z[:,0] < 0) & (spec_z[:,2] > 0)] = 1  # wake

    if not exclude_rem:
        states[(spec_z[:,1] > (spec_z[:,1].max()/4)) & (spec_z[:,0] < 0) & (spec_z[:,3] < -0.1)] = 3  # rem

    states[spec_z[:,3] > 1] = 0  # active

    state_mat = np.zeros((spec_z.shape[0], n_states), dtype=int)
    state_mat[np.arange(spec_z.shape[0]), states] = 1

    # create xarray of states
    states = xr.DataArray(state_mat, dims=['time', 'state'], 
                          coords={'time': spec.time, 'state': state_names})

    return states, spec_z


def add_states_xr(data_xr, states_xr, method='nearest'):
    """
    Add behavioral states to an existing xarray DataArray by aligning time points.

    Parameters
    ----------
    data_xr : xarray.DataArray
        Original data array with a 'time' dimension.
    states_xr : xarray.DataArray
        States array with dimensions (time, state) and state names in states.state.
    method : str, optional
        Interpolation method for aligning time points (default is 'nearest').

    Returns
    -------
    data_xr_with_states : xarray.DataArray
        Original data array with an added 'states' coordinate.
    """

    # Interpolate states to match data_xr time points
    interp_states = states_xr.interp(time=data_xr.time, method=method)

    # turn states into array of strings indicating current state
    state_vals = interp_states.state.values

    interp_states = xr.DataArray(
        np.array([state_vals[np.where(interp_states.isel(time=i).values==1)[0][0]] 
                  for i in range(interp_states.time.size)]),
        dims=['time'],
        coords={'time': data_xr.time}
    )
    
    # Add states as a new coordinate
    data_xr = data_xr.assign_coords(state=interp_states)

    return data_xr


def plot_band_power_states(band_power, states, band_names=None, colors=None,
                           axs=None):
    """
    Plot band power (time x band) and overlay sleep states.

    Parameters
    ----------
    band_power : np.ndarray or xarray.DataArray
        Matrix of shape (time, band) containing z-score band power values.
    states : xarray.DataArray
        States array with shape (time, state) and state names in states.state.
    band_names : list of str, optional
        Names for each band (for legend).
    colors : dict, optional
        Mapping from state name to color.
    axs : 2 axes
        Axes to plot state data in. First is for z-score of power,
        second is for state classifications.
    """

    if band_names is None:
        band_names = [f'band{i+1}' for i in range(band_power.shape[1])]
    if colors is None:
        colors = {'active': 'red', 'wake': 'blue', 'nrem': 'purple', 'rem': 'black'}

    if axs is None:
        fig, axs = plt.subplots(2,1,figsize=(15, 4), sharex=True)
        
    time = states.time.values

    for i, name in enumerate(band_names):
        axs[0].plot(time, band_power[:, i], label=name)
    axs[0].legend()

    for i, state_name in enumerate(states.state.values):
        axs[1].fill_between(
            time,
            0, i+1,
            where=states.sel(state=state_name).values.astype(bool),
            color=colors.get(state_name, 'gray'),
            alpha=0.3,
            label=state_name
        )
    axs[1].legend()
    axs[1].set_xlim(time[0], time[-1])
    axs[1].grid()
    
    return axs