"""Load dclut LFP files
Author: Drew B. Headley
"""

import numpy as np
import scipy.signal as ss
import dclut as dcl
from mimo_pack.analysis.probe import nearest_grid

def load_lfp_xr(lfp_path, notch_filter=False, remove_nan_time=True,
                dx=250, dy=100, notch_freq=60, notch_width=10):
    """
    Load LFP data from a dclut file as an xarray object, with options 
    for 60 Hz notch filtering, removing NaN time steps, and selecting 
    a grid of channels.

    Parameters
    ----------
    lfp_path : str
        Path to the dclut LFP file.
    notch_filter : bool, optional
        Whether to apply a 60 Hz notch filter (default: False).
    remove_nan_time : bool, optional
        Whether to remove time steps with NaN values (default: True).
    dx : float, optional
        Grid spacing in microns along x (default: 250).
    dy : float, optional
        Grid spacing in microns along y (default: 100).
    notch_freq : float, optional
        Frequency to notch filter (default: 60).
    notch_width : float, optional
        Notch filter width (default: 10).

    Returns
    -------
    lfp : xarray.DataArray
        LFP data as an xarray object.
    """
    # Load dclut object
    lfp_dcl = dcl.dclut(lfp_path)

    # Select grid of channels
    ch_grid = nearest_grid(lfp_dcl, dx=dx, dy=dy)[0]
    lfp_dcl.reset()
    lfp_dcl.points(select={'channel': ch_grid})
    lfp = lfp_dcl.read(format='xarray')[0]
    lfp = lfp.sortby(['ch_x', 'ch_y'])
    fs = 1/np.nanmedian(np.diff(lfp.time.to_numpy().flatten()))
    lfp = lfp.assign_attrs(sample_rate = fs)

    # Remove time steps with NaN if requested
    if remove_nan_time:
        mask = ~np.isnan(lfp.time.values)
        lfp = lfp.isel(time=mask)

    # Notch filter if requested
    if notch_filter:
        fs = 1 / np.nanmedian(np.diff(lfp.time.values))
        b, a = ss.iirnotch(notch_freq, notch_width, fs=fs)
        lfp.data = ss.filtfilt(b, a, lfp.values, axis=0)

    return lfp