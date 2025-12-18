"""Load dclut nidq files
Author: Drew B. Headley
"""

import numpy as np
import scipy.signal as ss
import dclut as dcl
import xarray as xr

def load_nidq_xr(nidq_path: str, chan_names: dict=None, remove_nan_time=True) -> xr.DataArray:
    """
    Load LFP data from a dclut file as an xarray object, with options 
    for 60 Hz notch filtering, removing NaN time steps, and selecting 
    a grid of channels.

    Parameters
    ----------
    lfp_path : str
        Path to the dclut LFP file.
    chan_names : dict, optional
        Dictionary mapping standard channel names to those in the file.
    remove_nan_time : bool, optional
        Whether to remove time steps with NaN values (default: True).
   

    Returns
    -------
    nidq : xarray.DataArray
        NIDQ data as an xarray object.
    """
    # Load dclut object
    nidq_dcl = dcl.dclut(nidq_path)

    num_t_samples = nidq_dcl.shape[1]

    nidq_dcl.reset()
    nidq_dcl.interval(select={'s1': [0, num_t_samples]})
    nidq_xr = nidq_dcl.read(format='xarray')[0]
    
    fs = 1/np.nanmedian(np.diff(nidq_xr.time.to_numpy().flatten()))
    nidq_xr = nidq_xr.assign_attrs(sample_rate = fs)
    
    # Remove time steps with NaN if requested
    if remove_nan_time:
        mask = ~np.isnan(nidq_xr.time.values)
        nidq_xr = nidq_xr.isel(time=mask)
    
    return nidq_xr