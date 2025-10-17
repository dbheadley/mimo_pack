# Analysis of behavioral states
# Authors: Drew B Headley
# Date: 2025-10-07

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import zscore
from mimo_pack.analysis.spectral import stft_xr, fit_spectrum_aperiodic_xr

def states_lfp_only(lfp, delta=[1, 4], theta=(6,9), high=(20,80), exclude_rem=False):
    """
    Simple state scoring based on LFP band power.
    Uses z-scored band power in delta, theta, and high gamma bands to classify
    time points into wake, nrem, and rem states.
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
    exclude_rem : bool, optional
        If True, only classify into wake and nrem states (default is False).
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

    if exclude_rem:
        n_states = 2
        state_names = ['wake', 'nrem']
    else:
        n_states = 3
        state_names = ['wake', 'nrem', 'rem']

    spec_z = zscore(np.vstack([delta_pow, theta_pow, high_pow]).T, axis=0)

    # wake: high gamma, low delta
    # nrem: high delta, low gamma
    # rem: high theta, low delta
    states = np.zeros(spec_z.shape[0], dtype=int)

    states[(spec_z[:,0] > 0) & (spec_z[:,2] < 0)] = 1  # nrem
    states[(spec_z[:,0] < 0) & (spec_z[:,2] > 0)] = 0  # wake
    if not exclude_rem:
        states[(spec_z[:,1] > (spec_z[:,1].max()/4)) & (spec_z[:,0] < 0)] = 2  # rem

    state_mat = np.zeros((spec_z.shape[0], n_states), dtype=int)
    state_mat[np.arange(spec_z.shape[0]), states] = 1

    # create xarray of states
    states = xr.DataArray(state_mat, dims=['time', 'state'], 
                          coords={'time': spec.time, 'state': state_names})

    return states, spec_z

def plot_band_power_states(band_power, states, band_names=None, colors=None,
                           ax=None):
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
    """

    if band_names is None:
        band_names = [f'band{i+1}' for i in range(band_power.shape[1])]
    if colors is None:
        colors = {'wake': 'red', 'nrem': 'purple', 'rem': 'black'}

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 4))
        
    time = states.time.values

    for i, name in enumerate(band_names):
        ax.plot(time, band_power[:, i], label=name)

    ax.set_xlim(time[0], time[-1])

    for i, state_name in enumerate(states.state.values):
        ax.fill_between(
            time,
            0, i+1,
            where=states.sel(state=state_name).values.astype(bool),
            color=colors.get(state_name, 'gray'),
            alpha=0.3,
            label=state_name
        )

    ax.grid()
    ax.legend()
    return ax