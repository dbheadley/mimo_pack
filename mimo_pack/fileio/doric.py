"""Load doric files
Author: Drew B. Headley
"""


import h5py
import numpy as np
import xarray as xr
from mimo_pack.util.hdf5 import get_h5_hierarchy

def load_doric_xr(doric_file_path: str) -> xr.DataArray:
    """
    Reads a Doric file and converts it into an xarray.DataArray.

    Parameters
    ----------
    doric_file_path: string

    Returns
    -------
    power_xr: xarray.DataArray
        An xarray.Data.Array containing the power output channels
        The data is organized into a 2D array with the dimensions:
        - time: The time points of the recording.
        - channel: The hierarchy of keys that led to the dataset.
    data_xr: xarray.DataArray
        An xarray.DataArray containing the recorded signals in the doric file.
        The data is organized into a 3D array with the dimensions:
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
    for (name, item_type, item_value) in data:
        if item_type == 'Dataset':
            data_series.append(item_value)
            hier_names.append(name)
            is_time.append(name.endswith('Time'))
            is_aout.append(name.find('AnalogOut') != -1)

    
    # Convert to numpy arrays
    hier_names = np.array(hier_names)
    is_time = np.array(is_time)
    is_aout = np.array(is_aout)
    # Shorten the hierarchy names to remove the common prefix
    hier_names = [name.split('Series0001/')[-1] for name in hier_names]

    # get indices for channel types
    pow_time_idx = np.where(is_time & is_aout)[0][0]
    pow_idx = np.where(~is_time & is_aout)[0]
    data_time_idx = np.where(is_time & ~is_aout)[0][0]
    data_idx = np.where(~is_time & ~is_aout)[0]

    # Create xarray DataArray from the list of datasets into a 2D numpy array, with
    # time x channel. Exclude datasets that end in 'Time'
    data_2d = np.array([data_series[i] for i in data_idx]).T
    data_ch = [hier_names[i] for i in data_idx]
    data_time_coord = np.array(data_series[data_time_idx])
    sample_rate_data = 1.0 / np.median(np.diff(data_time_coord))    


    # Create the xarray DataArray
    # We provide the 3D numpy array, the dimension names, and the coordinate arrays
    data_xr = xr.DataArray(
        data_2d,
        dims=("time", "channel"),
        coords={
            "time": data_time_coord,
            "channel": data_ch
        }
    )
    
    # Add attributes to the DataArray for metadata
    data_xr.attrs['doric_file_path'] = doric_file_path
    data_xr.attrs['sample_rate'] = sample_rate_data
    data_xr.time.attrs['units'] = 's'

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
    

    return data_xr, power_xr