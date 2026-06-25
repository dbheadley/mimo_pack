# Analysis function for AP data

from mimo_pack.fileio.ap import load_ap_windows_xr
from mimo_pack.util.xarray import interp_stack_xr
from mimo_pack.analysis.signal import envelope_xr
import numpy as np

def evoked_ap(ap_path, times, pre=0.01, post=0.03, spacing=20, clip_min=5, clip_max=20):
    """Calculate the mean evoked action potential envelope from raw data.

    This function loads windows of action potential (AP) data centered around
    specified event times, interpolates the data spatially, computes the mean
    evoked potential across all windows, calculates the temporal envelope of this
    mean signal, and finally clips the envelope to a specified amplitude range.

    Args:
        ap_path (str): 
            Path to the action potential data file (e.g., SpikeGLX .bin).
            times (array-like): A sequence of timestamps (in seconds) to center the
            extraction windows on.
        pre (float, optional): 
            The duration in seconds to include before each event time. Defaults to 0.01.
        post (float, optional): 
            The duration in seconds to include after each event time. Defaults to 0.03.
        spacing (int, optional): 
            The spatial interpolation spacing, likely in microns. Defaults to 20.
        clip_min (int, optional): 
            The minimum value to clip the final envelope amplitude to. Defaults to 5.
        clip_max (int, optional): 
            The maximum value to clip the final envelope amplitude to. Defaults to 20.

    Returns:
        xarray.DataArray: A DataArray containing the clipped envelope of the mean
            evoked action potential. The time coordinate is converted to
            milliseconds.
    """
    ap_data = load_ap_windows_xr(ap_path, centers=times, pre=pre, post=post)
    ap_interp = interp_stack_xr(ap_data, spacing=spacing)
    ap_mean = ap_interp.mean(dim='window')
    ap_mean = ap_mean.assign_coords({'time': ap_mean['time'].values * 1e3})  # convert to ms
    ap_env = envelope_xr(ap_mean, dim='time')
    ap_env = np.clip(ap_env, clip_min, clip_max)
    return ap_env