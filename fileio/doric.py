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


def doric_to_binary_dclut(doric_file_path: str, out_dir: str = None, dtype: str = 'float32') -> str:
    """
    Convert a Doric file to a binary file containing all traces and a dclut JSON metadata file.

    Parameters
    ----------
    doric_file_path : str
        Path to the Doric file (.h5).
    out_dir : str, optional
        Output directory for binary and dclut files. If None, uses Doric file directory.
    dtype : str, optional
        Data type for binary file. Default is 'float32'.

    Returns
    -------
    bin_path : str
        Path to the binary file.
    dclut_path : str
        Path to the dclut JSON metadata file.
    """
    # Load Doric data
    fp_xr, power_xr, dio_xr = load_doric_xr(doric_file_path)

    # Stack all traces along channel axis
    # Align time axes by interpolating to the fp_xr time points
    time = fp_xr.time.values
    traces = [fp_xr.values]
    ch_names = list(fp_xr.channel.values)

    # Interpolate power_xr and dio_xr to fp_xr time axis if needed
    for xr_data in [power_xr, dio_xr]:
        # Interpolate if time axes differ
        if not np.array_equal(xr_data.time.values, time):
            interp = np.empty((len(time), xr_data.shape[1]), dtype=xr_data.dtype)
            for i in range(xr_data.shape[1]):
                interp[:, i] = np.interp(time, xr_data.time.values, xr_data[:, i])
            traces.append(interp)
        else:
            traces.append(xr_data.values)
        ch_names.extend(list(xr_data.channel.values))

    # Concatenate all traces (time x channel)
    all_traces = np.concatenate(traces, axis=1)
    all_traces = all_traces.astype(dtype)

    # Output paths
    if out_dir is None:
        out_dir = os.path.dirname(doric_file_path)
    base_name = os.path.splitext(os.path.basename(doric_file_path))[0]
    bin_path = os.path.join(out_dir, base_name + '_alltraces.bin')
    dclut_path = os.path.join(out_dir, base_name + '_alltraces_dclut.json')

    # Write binary file
    all_traces.tofile(bin_path)

    # Create dclut metadata
    shape = (all_traces.shape[0], all_traces.shape[1])  # (time, channel)
    scales = [
        {'name': 'time', 'dim': 0, 'unit': 's', 'type': 'list', 'val': time},
        {'name': 'channel', 'dim': 1, 'unit': 'au', 'type': 'list', 'val': ch_names}
    ]
    create_dclut(bin_path, shape, dcl_path=dclut_path, dtype=dtype, 
                 data_name='doric_traces', data_unit='au', scales=scales)

    return bin_path, dclut_path