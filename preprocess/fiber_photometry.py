"""Process and correct fiber photometry data
Created by: Drew Headley
"""

import xarray as xr
import numpy as np
from mimo_pack.analysis.filter import lp_filter_xr

def correct_fp_signal(raw: xr.DataArray, 
                      ch_groups: dict = {'ch1': ('LockInAOUT02/AIN01', 'LockInAOUT01/AIN01'),
                                         'ch2': ('LockInAOUT04/AIN02', 'LockInAOUT03/AIN02')},
                      z_window: int = None, cutoff_freq: float = 5) -> xr.DataArray:
    """
    Corrects the fluorescence photometry signal by regressing out movement from
    the isosbestic channel. 

    Parameters
    ----------
    raw : xr.DataArray
        The raw fluorescence photometry signal to be corrected. Must have a 'channel'
        and 'time' dimension.
    ch_groups : dictionary of tuples, optional
        A dictionary of tuples, where key is a new corrected channel to create,
        and whos value contains a tuple of names of the channels
        to be processed for that channel. Each tuple can be 1 or 2 elements, 
        with the first the signal channel and the second optional one the isosbestic channel,
        for movement correction. 
        Default is {'ch1': ('LockInAOUT02/AIN01', 'LockInAOUT01/AIN01'),
                    'ch2': ('LockInAOUT03/AIN02', 'LockInAOUT04/AIN02')}.
    z_window : numeric, optional
        The size of the moving window in seconds for z-score normalization.
        Default is no rolling window, z-score normalization is applied to the entire signal.
    cutoff_freq : float, optional
        The cutoff frequency for the low-pass filter in Hz.
        Default is 5 Hz.

    Returns
    -------
    pro : xr.DataArray
        The processed fluorescence photometry signal.
    """

    s_rate = raw.attrs.get('sample_rate', 1.0)

    # low-pass filter the signals
    raw_lp = lp_filter_xr(raw, cutoff_freq=cutoff_freq)
    
    df_f0 = []
    for _, ch_pair in ch_groups.items():
        if len(ch_pair) == 2:
            sig_ch = ch_pair[0]
            isos_ch = ch_pair[1]

            # Check if both channels exist in the signal
            if sig_ch not in raw_lp.channel.values or isos_ch not in raw_lp.channel.values:
                raise ValueError(f"Channels {sig_ch} or {isos_ch} not found in the signal.")
            
            # Get the signal and isosbestic channel data
            sig_data = raw_lp.sel(channel=sig_ch).to_numpy()
            isos_data = raw_lp.sel(channel=isos_ch).to_numpy()

            # Perform rolling linear regression
            x = np.stack((isos_data, np.ones_like(isos_data)), axis=1)  # Design matrix for regression
            slope, offset = np.linalg.lstsq(x, sig_data, rcond=None)[0]
            fitted = slope * isos_data + offset
            resid = sig_data - fitted

            # calculate df/f0
            df_f0.append((resid)/fitted)
            
    coords = raw_lp.coords.copy()
    coords['channel'] = [*ch_groups.keys()]

    # Remove coordinates associated with the original channels
    raw_lp = raw_lp.drop_vars('channel')

    # Create a new DataArray for the processed signal
    pro = xr.DataArray(
        np.array(df_f0).T,
        dims=raw_lp.dims,
        coords=coords,
        attrs=raw_lp.attrs
    )

    # Normalize the corrected signal using z-score normalization
    if z_window is not None:
        z_window = int(z_window * s_rate)
        pro_mean = pro.rolling(time=z_window, center=True).mean()
        pro = (pro - pro_mean) / pro.rolling(time=z_window, center=True).std()
    else:
        pro = (pro - pro.mean(dim='time')) / pro.std(dim='time')

    return pro
