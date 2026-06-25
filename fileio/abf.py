"""Load abf files
Author: Drew B. Headley
"""


import pyabf
import numpy as np
import xarray as xr


def load_abf_settings(abf_file_path: str) -> dict:
    """
    Reads the settings from an Axon Binary File (ABF) and returns them as a dictionary.

    Parameters
    ----------
    abf_file_path: string
        The full path to the .abf file.

    Returns
    -------
    settings: dict
        A dictionary containing the ABF settings, including protocol, number of sweeps,
        channel names, and units.
    """
    abf = pyabf.ABF(abf_file_path)
    
    settings = {
        'protocol': abf.protocol,
        'n_sweeps': abf.sweepCount,
        'channel_names': abf.adcNames,
        'channel_units': abf.adcUnits,
        'sample_rate': abf.sampleRate,
        'created': abf.abfDateTime
    }
    
    return settings

def load_abf_xr(abf_file_path: str) -> xr.DataArray:
    """
    Reads an Axon Binary File (ABF) and converts it into an xarray.DataArray.

    Parameters
    ----------
    abf_file_path: string
        The full path to the .abf file.

    Returns
    -------
    data_xr: xarray.DataArray    
        An xarray.DataArray containing the ABF data.
        The data is organized into a 3D array with the dimensions:
        - time: The time points of the recording.
        - channel: The recording channel.
        - trial: The sweep or trial number.

    Initial code generated with Gemini
    """
    # Load the ABF file using the pyabf library
    abf = pyabf.ABF(abf_file_path)

    # Get the dimensions of the data from the ABF header
    n_sweeps = abf.sweepCount
    n_channels = abf.channelCount
    n_points = abf.sweepPointCount

    # Create an empty 3D numpy array to store all the data
    # The dimensions are ordered as (trial, channel, time)
    all_data = np.zeros((n_points, n_channels, n_sweeps))

    # Iterate over each sweep (trial) and channel to fill the numpy array
    for sweep_number in range(n_sweeps):
        for channel_number in range(n_channels):
            # Set the sweep and channel to read the data
            abf.setSweep(sweepNumber=sweep_number, channel=channel_number)
            # Store the data in the corresponding location in the 3D array
            all_data[:, channel_number, sweep_number] = abf.sweepY

    # Create the coordinate arrays for the xarray DataArray
    # The time coordinates are taken from the last sweep (they are the same for all sweeps)
    time_coords = abf.sweepX
    # The channel coordinates are a simple range of integers
    channel_coords = np.arange(n_channels)
    # The trial coordinates are also a simple range of integers
    trial_coords = np.arange(n_sweeps)
    # Channel names
    names_coords = abf.adcNames
    # Channel units
    units_coords = abf.adcUnits

    # Create the xarray DataArray
    # We provide the 3D numpy array, the dimension names, and the coordinate arrays
    data_xr = xr.DataArray(
        all_data,
        dims=("time", "channel", "trial"),
        coords={
            "time": time_coords,
            "channel": channel_coords,
            "trial": trial_coords,
            "ch_names": ('channel', names_coords),
            "ch_units": ('channel', units_coords)
        }
    )
    
    # Add attributes to the DataArray for metadata
    data_xr.attrs['abf_file_path'] = abf_file_path
    data_xr.attrs['abf_version'] = abf.abfVersionString
    data_xr.attrs['protocol'] = abf.protocol
    data_xr.attrs['sample_rate'] = abf.sampleRate
    data_xr.attrs['created'] = abf.abfDateTime
    data_xr.time.attrs['units'] = 's'

    return data_xr

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    try:
        abf_file_path = os.path.join('..','..','test_data','invitro','21520000.abf')
        
        # Convert the ABF file to an xarray DataArray
        abf_xr = load_abf_xr(abf_file_path)

        # Print the resulting xarray DataArray to see its structure
        print("Successfully converted ABF to xarray.DataArray:")
        print(abf_xr)
        
        volt_chans = np.where(abf_xr.ch_units.values == 'mV')[0]
        fig, ax = plt.subplots(volt_chans.size,1, sharex=True, sharey=True)
        for i, ch in enumerate(volt_chans):
            ax[i].plot(abf_xr.time, abf_xr.isel(channel=ch), color='k')
            ax[i].set_title(f'Channel {ch} - {abf_xr.ch_names.values[ch]}')
            ax[i].set_ylabel(abf_xr.ch_units.values[ch])
            ax[i].set_xlabel(f'Time ({abf_xr.time.attrs["units"]})')

        fig.tight_layout()
        plt.show()
        
        
    except Exception as e:
        print(f"Error details: {e}")