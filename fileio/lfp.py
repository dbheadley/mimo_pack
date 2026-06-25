"""Load dclut LFP files
Author: Drew B. Headley
Refactored to match ap.py functionality with LFP-specific defaults.
"""

import numpy as np
import scipy.signal as ss
import dclut as dcl
import xarray as xr
from mimo_pack.analysis.probe import nearest_grid

# ==========================================
# Helper Functions (Internal)
# ==========================================

def _configure_channels(lfp_dcl, dx, dy, chans):
    """
    Selects channels based on input or grid spacing.
    """
    if chans is None:
        chans = nearest_grid(lfp_dcl, dx=dx, dy=dy)[0]
    
    lfp_dcl.points(select={'channel': chans})
    return chans

def _filter_array(data, fs, notch_freq, notch_width, lfp_band):
    """
    Applies Notch and Bandpass filters to a numpy array along axis 0.
    """
    # Notch filter
    if notch_freq is not None:
        b, a = ss.iirnotch(notch_freq, notch_width, fs=fs)
        data = ss.filtfilt(b, a, data, axis=0)

    # Bandpass filter
    if lfp_band is not None:
        b, a = ss.butter(3, [lfp_band[0]/(fs/2), lfp_band[1]/(fs/2)], btype='bandpass')
        data = ss.filtfilt(b, a, data, axis=0)
        
    return data

def _post_process(xr_obj, fs, notch_freq, notch_width, lfp_band):
    """
    Applies filters and attributes to an xarray object (or list of them).
    """
    def process_single(x_data):
        x_data.data = _filter_array(x_data.values, fs, notch_freq, notch_width, lfp_band)
        return x_data.assign_attrs(sample_rate=fs)

    if isinstance(xr_obj, list):
        return [process_single(x) for x in xr_obj]
    else:
        return process_single(xr_obj)

def _validate_intervals(dcl_obj, intervals):
    """
    Validates that requested intervals are within the dclut object's time range.
    """
    time_vals = dcl_obj.scale_values('time')
    t_min, t_max = np.nanmin(time_vals), np.nanmax(time_vals)
    delete_idxs = []
    for idx in range(intervals.shape[0]):
        if intervals[idx, 0] >= t_min and intervals[idx, 1] <= t_max:
            delete_idxs.append(idx)
    
    if len(delete_idxs) > 0:
        print(f"Warning: Had to remove {len(delete_idxs)} intervals outside data time range.")

    return np.delete(intervals, delete_idxs, axis=0)
            
# ==========================================
# Main Loading Functions
# ==========================================

def load_lfp_full_xr(lfp_path, lfp_band=None, remove_nan_time=True,
                     dx=250, dy=100, notch_freq=None, notch_width=10, chans=None):
    """
    Load entire time course of LFP data from a dclut file as an xarray object.
    
    Defaults are tailored for LFP:
    - dx/dy: 250/100 microns (coarser grid than AP)
    - notch_freq: 60 Hz (standard mains noise removal)
    - lfp_band: 0.5 - 300 Hz (standard LFP range)
    """
    lfp_dcl = dcl.dclut(lfp_path)
    lfp_dcl.reset()
    
    # Configure channels
    _configure_channels(lfp_dcl, dx, dy, chans)

    # Read data
    lfp_xr = lfp_dcl.read(format='xarray')[0]
    lfp_xr = lfp_xr.sortby(['ch_x', 'ch_y'])
    
    # Calculate FS
    fs = 1/np.nanmedian(np.diff(lfp_xr.time.to_numpy().flatten()))

    # Remove time steps with NaN if requested
    if remove_nan_time:
        mask = ~np.isnan(lfp_xr.time.values)
        lfp_xr = lfp_xr.isel(time=mask)

    # Apply filters and attributes
    lfp_xr = _post_process(lfp_xr, fs, notch_freq, notch_width, lfp_band)

    return lfp_xr


def load_lfp_intervals_xr(lfp_path, intervals, lfp_band=None, remove_nan_time=True,
                          dx=250, dy=100, notch_freq=None, notch_width=10, chans=None):
    """
    Load time intervals of LFP data from a dclut file as an xarray object.
    """
    lfp_dcl = dcl.dclut(lfp_path)
    lfp_dcl.reset()

    # Configure channels
    _configure_channels(lfp_dcl, dx, dy, chans)

    # Validate intervals
    intervals = _validate_intervals(lfp_dcl, intervals)

    # Select intervals
    lfp_dcl.intervals(select={'time': intervals}, select_mode='split')
    
    # Read data
    lfp_xr_list = lfp_dcl.read(format='xarray')
    lfp_xr_list = [lfp.sortby(['ch_x', 'ch_y']) for lfp in lfp_xr_list]
    
    # Calculate FS (from first interval)
    fs = 1/np.nanmedian(np.diff(lfp_xr_list[0].time.to_numpy().flatten()))

    # Remove time steps with NaN if requested
    if remove_nan_time:
        lfp_xr_list = [lfp.isel(time=~np.isnan(lfp.time.values)) for lfp in lfp_xr_list]

    # Apply filters and attributes
    lfp_xr_list = _post_process(lfp_xr_list, fs, notch_freq, notch_width, lfp_band)

    return lfp_xr_list


def load_lfp_windows_xr(lfp_path, centers, pre, post, lfp_band=None, remove_nan_time=True,
                        dx=250, dy=100, notch_freq=None, notch_width=10, chans=None):
    """
    Load time windows of LFP data (e.g., for ERPs) from a dclut file as an xarray object.
    """
    lfp_dcl = dcl.dclut(lfp_path)
    
    # Calculate FS from raw time scale
    times = lfp_dcl.scale_values('time')
    fs = int(1/np.nanmedian(np.diff(times)))

    # Calculate center indices and intervals
    center_idxs = np.array([np.nanargmin(np.abs(times - c)) for c in centers])
    pre_samples = int(pre * fs)
    post_samples = int(post * fs)
    intervals = np.column_stack((center_idxs - pre_samples,
                                 center_idxs + post_samples))

    # Configure channels
    lfp_dcl.reset()
    _configure_channels(lfp_dcl, dx, dy, chans)

    # Validate intervals
    intervals = _validate_intervals(lfp_dcl, intervals)
    
    # Select intervals based on sample indices
    lfp_dcl.intervals(select={'s0': intervals}, select_mode='split')
    lfp_xr_list = lfp_dcl.read(format='xarray')

    # Remove windows that overlapped with NaN times entirely
    if remove_nan_time:
        nan_intervals = np.where(~np.any(np.isnan(times[intervals]), axis=1))[0]
        lfp_xr_list = [lfp_xr_list[i] for i in nan_intervals]
        # Adjust centers if windows were dropped
        centers = centers[nan_intervals] if len(centers) == len(intervals) else centers

    # Standardize time coordinates to be relative to center
    relative_inds = np.arange(-pre_samples, post_samples)
    relative_times = relative_inds / fs
    
    for i in range(len(lfp_xr_list)):
        lfp_xr_list[i] = lfp_xr_list[i].assign_coords({
            'time': relative_times,
            's0': ('time', relative_inds)
        })

    # Concatenate
    lfp_xr = xr.concat(lfp_xr_list, dim='window').transpose('time', 'channel', 'window')
    lfp_xr = lfp_xr.sortby(['ch_x', 'ch_y'])
    lfp_xr = lfp_xr.assign_coords({'window': centers})

    # Apply filters and attributes
    lfp_xr = _post_process(lfp_xr, fs, notch_freq, notch_width, lfp_band)

    return lfp_xr