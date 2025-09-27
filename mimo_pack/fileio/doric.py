"""Load doric files
Author: Drew B. Headley
"""


import h5py
import numpy as np
import xarray as xr
from mimo_pack.util.hdf5 import get_h5_hierarchy
from scipy.interpolate import interp1d

def load_doric_xr(doric_file_path: str) -> xr.DataArray:
    """
    Reads a Doric file and converts it into an xarray.DataArray.

    Parameters
    ----------
    doric_file_path: string

    Returns
    -------
    fp_xr: xarray.DataArray
        An xarray.DataArray containing the fiber photometry signals in the doric file.
        The data is organized into a 2D array with the dimensions:
        - time: The time points of the recording.
        - channel: The hierarchy of keys that led to the dataset.
    power_xr: xarray.DataArray
        An xarray.Data.Array containing the power output channels
        The data is organized into a 2D array with the dimensions:
        - time: The time points of the recording.
        - channel: The hierarchy of keys that led to the dataset.
    dio_xr: xarray.DataArray
        An xarray.DataArray containing the digital output signals in the doric file.
        The data is organized into a 2D array with the dimensions:
        - time: The time points of the recording.
        - channel: The hierarchy of keys that led to the dataset.

    Initial code generated with Gemini
    """
    # Load the Doric file using the h5py library
    data = h5py.File(doric_file_path, 'r')

    # Extract the relevant datasets from the Doric file
    data = get_h5_hierarchy(data)

    data_series = []
    hier_names = []
    is_time = []
    is_aout = []
    is_dio = []
    is_fp = []
    for (name, item_type, item_value) in data:
        if item_type == 'Dataset':
            data_series.append(item_value)
            hier_names.append(name)
            is_time.append(name.endswith('Time'))
            is_fp.append(name.find('LockInAOUT') != -1)
            is_aout.append(name.find('AnalogOut') != -1)
            is_dio.append(name.find('DigitalIO') != -1)


    
    # Convert to numpy arrays
    hier_names = np.array(hier_names)
    is_time = np.array(is_time)
    is_fp = np.array(is_fp)
    is_aout = np.array(is_aout)
    is_dio = np.array(is_dio)

    # Shorten the hierarchy names to remove the common prefix
    hier_names = [name.split('Series0001/')[-1] for name in hier_names]

    # get indices for channel types
    pow_time_idx = np.where(is_time & is_aout)[0][0]
    pow_idx = np.where(~is_time & is_aout)[0]
    fp_time_idx = np.where(is_time & is_fp)[0][0]
    fp_idx = np.where(~is_time & is_fp)[0]
    dio_time_idx = np.where(is_time & is_dio)[0][0]
    dio_idx = np.where(~is_time & is_dio)[0]

    # Create xarray DataArray from the list of datasets into a 2D numpy array, with
    # time x channel. Exclude datasets that end in 'Time'
    fp_2d = np.array([data_series[i] for i in fp_idx]).T
    fp_ch = [hier_names[i] for i in fp_idx]
    fp_time_coord = np.array(data_series[fp_time_idx])
    sample_rate_fp = 1.0 / np.median(np.diff(fp_time_coord))


    # Create the xarray DataArray
    # We provide the 3D numpy array, the dimension names, and the coordinate arrays
    fp_xr = xr.DataArray(
        fp_2d,
        dims=("time", "channel"),
        coords={
            "time": fp_time_coord,
            "channel": fp_ch
        }
    )
    
    # Add attributes to the DataArray for metadata
    fp_xr.attrs['doric_file_path'] = doric_file_path
    fp_xr.attrs['sample_rate'] = sample_rate_fp
    fp_xr.time.attrs['units'] = 's'

    # Extract the power output channels
    power_2d = np.array([data_series[i] for i in pow_idx]).T
    power_ch = [hier_names[i] for i in pow_idx]
    power_time_coord =  np.array(data_series[pow_time_idx])
    sample_rate_power = 1.0 / np.median(np.diff(power_time_coord))

    # Create the xarray DataArray for power output channels
    power_xr = xr.DataArray(
        power_2d,
        dims=("time", "channel"),
        coords={
            "time": power_time_coord,
            "channel": power_ch
        }
    )
    # Add attributes to the power DataArray for metadata
    power_xr.attrs['doric_file_path'] = doric_file_path
    power_xr.attrs['sample_rate'] = sample_rate_power
    power_xr.time.attrs['units'] = 's'
    
    # Extract the digital IO channels
    dio_2d = np.array([data_series[i] for i in dio_idx]).T
    dio_ch = [hier_names[i] for i in dio_idx]
    dio_time_coord = np.array(data_series[dio_time_idx])
    sample_rate_dio = 1.0 / np.median(np.diff(dio_time_coord))

    # Resample digital IO to match data_xr time points
    # Use nearest neighbor interpolation for digital signals
    interp_func = interp1d(
        dio_time_coord, dio_2d, axis=0, kind='nearest', bounds_error=False, fill_value=(dio_2d[0], dio_2d[-1])
    )
    dio_2d_resampled = interp_func(fp_xr.time.values)
    dio_time_resampled = fp_xr.time.values

    # Create the xarray DataArray for digital IO channels
    dio_xr = xr.DataArray(
        dio_2d_resampled,
        dims=("time", "channel"),
        coords={
            "time": dio_time_resampled,
            "channel": dio_ch
        }
    )
    # Add attributes to the digital IO DataArray for metadata
    dio_xr.attrs['doric_file_path'] = doric_file_path
    dio_xr.attrs['sample_rate'] = sample_rate_dio
    dio_xr.time.attrs['units'] = 's'

    return fp_xr, power_xr, dio_xr