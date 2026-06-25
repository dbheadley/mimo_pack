# Analysis function for LFP data

from mimo_pack.fileio.lfp import load_lfp_windows_xr
from mimo_pack.util.xarray import interp_stack_xr
from mimo_pack.analysis.csd import csd_xr
import numpy as np

def evoked_lfp(lfp_path, times, pre=0.01, post=0.03, spacing=100):
    """Calculate the event-triggered average Local Field Potential (LFP) and its CSD.

    This function loads LFP data from a specified path, extracts windows around
    given event times, performs spatial interpolation, and computes the mean LFP
    across all windows. The resulting evoked LFP is baseline-corrected using the
    median of the pre-event period. Finally, it calculates the Current Source
    Density (CSD) from the mean LFP.

    Parameters
    ----------
    lfp_path : str or path-like
        Path to the LFP data file.
    times : array-like
        An array of event timestamps (in seconds) to align the LFP data to.
    pre : float, optional
        The time in seconds to include before each event time. Defaults to 0.01.
    post : float, optional
        The time in seconds to include after each event time. Defaults to 0.03.
    spacing : int, optional
        The desired spatial spacing between channels for interpolation (e.g., in
        micrometers). Defaults to 100.

    Returns
    -------
    lfp_mean : xarray.DataArray
        The event-triggered average LFP, baseline-corrected. The time coordinate
        is converted to milliseconds.
    csd_mean : xarray.DataArray
        The Current Source Density calculated from `lfp_mean`.

    See Also
    --------
    load_lfp_windows_xr : Loads LFP data in windows around specified times.
    interp_stack_xr : Interpolates a stack of LFP data.
    csd_xr : Calculates the Current Source Density.

    """
    lfp_data = load_lfp_windows_xr(lfp_path, centers=times, pre=pre, post=post)
    lfp_interp = interp_stack_xr(lfp_data, spacing=spacing)
    lfp_mean = lfp_interp.mean(dim='window')    
    lfp_mean = lfp_mean - lfp_mean.sel(time=slice(-pre,0)).median(dim='time')
    lfp_mean = lfp_mean.assign_coords(time=lfp_mean.time * 1e3)  # convert to ms
    csd_mean = csd_xr(lfp_mean, coord='ch_y')
    return lfp_mean, csd_mean

