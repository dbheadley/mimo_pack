# LFP preprocessing related functions
# Author: Drew Headley
# Created: 2024-05-19

import os
import shutil
import numpy as np
import pandas as pd
import dclut as dcl
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from dclut import create_dclut
from ..fileio.spikeglx import read_meta, get_chanmap, get_geommap
from ..math.curvefitting import piecewise_table_monotonic

def make_lfp_file_dclut(dcl_path, lfp_path, lfp_cutoff=200, lfp_fs=1000, sync = {'channel': [384]},
                        time_scale='time', verbose=False):
    """
    Makes LFP file from binary with a dclut metadata file. Overwrites existing LFP file.

    Parameters
    ----------
    dcl_path : str
        Path to dclut file.
    lfp_path : str
        Path to use when saving the LFP binary file. The corresponding dclut file will be saved with
        the same name but with a _dclut.json extension.

    Optional
    --------
    lfp_cutoff : numeric
        Cutoff frequency for low-pass filter in Hz. Default is 200 Hz.
    lfp_fs : numeric
        Sampling frequency of LFP data in Hz. Default is 1000 Hz.
    sync : dict
        Dictionary that can be passed to dclut with the channel number 
        for the sync signal. If no sync signal is present, give as
        None. Default is {'channel': [384]}.
    time_scale : str
        Name of the time scale in the dclut file. Default is 'time'.
    verbose : bool
        If True, print progress. Default is False.

    Returns
    -------
    lfp_dcl_path : str
        Path to LFP dclut file.
    """

    # Check if the binary file exists
    if not os.path.exists(dcl_path):
        raise FileNotFoundError('Dclut file {} not found'.format(dcl_path))
    
    if sync is None:
        sync_chan = []
    else:
        sync_chan = sync['channel']

    # Create dclut object for reading binary data
    dcl_data = dcl.dclut(dcl_path)
    times = dcl_data.scale_values(time_scale)
    time_dim = dcl_data.dcl['scales'][time_scale]['dim']
    ind_scale = 's' + str(time_dim)
    ind_len = dcl_data.shape[time_dim]

    # determine which time points to keep for downsampling
    fs = np.round(1/np.nanmedian(np.diff(times))).astype(int)
    keep_step = int(fs/lfp_fs)
    down_inds = np.arange(0, ind_len, keep_step)
    chunks = range(0, ind_len, fs*60)
    if chunks[-1] != ind_len:
        chunks = np.append(chunks, ind_len)
    
    # initialize binary data to write
    lfpf = open(lfp_path, mode='wb')

    if verbose:
        iter_chunks = tqdm(range(len(chunks)-1))
    else:
        iter_chunks = range(len(chunks)-1)
    
    # iterate through chunks and write out to LFP file
    lfp_times = []
    for i in iter_chunks:
        # read in chunk of data
        dcl_data.reset()

        # include a buffer of 1sec time points on either side of the chunk
        # to minimize filtering edge effects
        if chunks[i]-fs > 0:
            low_edge = fs
        else:
            low_edge = 0
        
        if chunks[i+1]+fs < ind_len:
            high_edge = fs
        else:
            high_edge = 0
        
        dcl_data.intervals({ind_scale: [chunks[i]-low_edge, chunks[i+1]+high_edge-1]})
        bin_data = dcl_data.read(format='xarray')[0]
        chunk_times = bin_data[time_scale].values
        chunk_inds = bin_data[ind_scale].values

        if len(chunk_times) > 10:
            # calculate LFP
            lfp_data = calc_lfp(bin_data.data, fs, lfp_cutoff=lfp_cutoff, 
                                down_factor=1, offset=0, ignore_chans=sync_chan)
            
            # remove buffer time points
            lfp_data = lfp_data[low_edge:-(high_edge-1),:]
            chunk_times = chunk_times[low_edge:-(high_edge-1)]
            chunk_inds = chunk_inds[low_edge:-(high_edge-1)]

            # lfp data to keep for downsampling
            down_bool = np.where(np.isin(chunk_inds, down_inds))[0]
            lfp_data = lfp_data[down_bool,:]
            lfp_times.append(chunk_times[down_bool])
            lfpf.write(lfp_data.tobytes())
    lfpf.close()

    # write new dclut file for the LFP by copying the original and changing the time fields
    lfp_dcl_path = lfp_path.replace('.bin', '_dclut.json')
    lfp_times = np.concatenate(lfp_times).flatten()
    dcl_data.reset()
    dcl_data.dcl['file']['name'] = lfp_path
    dcl_data.dcl['file']['shape'][time_dim] = len(lfp_times)
    dcl_data.dcl['scales'][time_scale]['type'] = 'table'
    dcl_data.dcl['scales'][time_scale]['values'] = piecewise_table_monotonic(lfp_times)
    dcl_data.save(path=lfp_dcl_path)
    
    return lfp_path

def make_lfp_file_spikeglx(bin_path, lfp_cutoff=200, lfp_fs=1000, suffix='lfp', verbose=False):
    """
    Makes LFP file from raw SpikeGLX binary. Overwrites existing LFP file.

    Parameters
    ----------
    bin_path : str
        Path to binary file.
    

    Optional
    --------
    lfp_cutoff : numeric
        Cutoff frequency for low-pass filter in Hz. Default is 200 Hz.
    lfp_fs : numeric
        Sampling frequency of LFP data in Hz. Default is 1000 Hz.
    suffix : str
        Suffix for LFP file. Default is 'lfp'.
    verbose : bool
        If True, print progress. Default is True.

    Returns
    -------
    lfp_path : str
        Path to LFP file.
    """

    # Check if the binary file exists
    if not os.path.exists(bin_path):
        raise FileNotFoundError('Binary file {} not found'.format(bin_path))
    
    # Get the meta data
    meta = read_meta(bin_path)

    fs = meta['imSampRate']
    chan_num = meta['nSavedChans']
    sy_yes = meta['acqApLfSy'][2]
    byte_num = 2

    if sy_yes:
        sy_chan = chan_num-1
    else:
        sy_chan = []

    # open binary file, read only
    binf = open(bin_path, mode='rb')
    bin_bytes = os.path.getsize(bin_path)

    # test if fs is evenly divisible by lfp_fs
    down_factor = fs/lfp_fs
    if down_factor % 1 != 0:
        raise ValueError('Sampling rate of {} Hz is not evenly divisible by ' 
                         'by LFP sampling rate of {} Hz'.format(fs, lfp_fs))
    else:
        down_factor = int(down_factor)

    chunk_dur = 60*fs # number of time points to convert at a time
    step_bytes = byte_num*chan_num # number of bytes per time point
    bin_dur = bin_bytes//step_bytes # number of time points in binary file

    # initialize binary data to write
    lfp_path = bin_path.replace('.ap.bin', '.{}.bin'.format(suffix))
    lfpf = open(lfp_path, mode='wb')

    # convert to LFP, in 1 minute chunks, 
    chunks = np.arange(0, bin_dur, chunk_dur).astype(np.int64)
    if chunks[-1] != bin_dur:
        chunks = np.append(chunks, bin_dur)

    # downsampled LFP time points to keep
    lfp_keep = np.arange(0, bin_dur, down_factor).astype(np.int64)

    if verbose:
        iter_chunks = tqdm(range(len(chunks)-1))
    else:
        iter_chunks = range(len(chunks)-1)
    
    # iterate through chunks
    for i in iter_chunks:
        # read in chunk of data
        binf.seek(chunks[i]*step_bytes)
        chunk_len = chunks[i+1]-chunks[i]
        bin_data = np.fromfile(binf, dtype='int16', count=chunk_len*chan_num)
        bin_data = bin_data.reshape((chunk_len, chan_num))

        # offset is the first index in lfp_keep that is greater than or 
        # equal to chunks[i]
        offset = lfp_keep[np.nonzero(lfp_keep >= chunks[i])[0][0]]-chunks[i]

        # calculate LFP
        lfp_data = calc_lfp(bin_data, fs, lfp_cutoff=lfp_cutoff, 
                            down_factor=down_factor, offset=offset, ignore_chans=sy_chan)
        
        lfpf.write(lfp_data.tobytes())

    binf.close()
    lfpf.close()
    return lfp_path

def dclut_from_meta_lfp(lfp_path, dcl_path=None, lfp_fs=1000):
    """
    Create a dclut json file from the .meta file associated with a SpikeGLX. bin file
    that has been converted to LFP.

    Parameters
    ----------
    lfp_path : str
        Path to the LFP binary file
    
    Optional
    --------
    dcl_path : str
        Path to save the dclut json file. If not provided, the file will be saved in
        the same directory as the binary file with the same name but with a _dclut.json extension.
    
    Returns
    -------
    dcl_path : str
        Path to the dclut json file
    """
    
    if dcl_path is None:
        dcl_path = lfp_path.replace('.bin', '_dclut.json')
    
    bin_path = lfp_path.replace('.lfp.bin', '.ap.bin')
    meta = read_meta(bin_path)
    chmap = get_chanmap(bin_path)
    
    chan_num = meta['nSavedChans']

    gmap = get_geommap(bin_path)
    chan_props = chmap.merge(gmap, left_index=True, right_index=True, how='outer')
    scales = [{'name': 'time', 'dim': 0, 'unit': 'seconds', 
                'type': 'linear', 'val': [1/lfp_fs, 0]}, 
                {'name': 'channel', 'dim': 1, 'unit': 'none', 
                'type': 'index', 'val': None}, 
                {'name': 'ch_name', 'dim': 1, 'unit': 'none', 
                'type': 'list', 'val': chan_props['name'].values}, 
                {'name': 'ch_order', 'dim': 1, 'unit': 'none', 
                'type': 'list', 'val': chan_props['order'].values}, 
                {'name': 'ch_x', 'dim': 1, 'unit': 'um', 
                'type': 'list', 'val': chan_props['x'].values}, 
                {'name': 'ch_y', 'dim': 1, 'unit': 'um', 
                'type': 'list', 'val': chan_props['y'].values}, 
                {'name': 'ch_shank', 'dim': 1, 'unit': 'none', 
                'type': 'list', 'val': chan_props['shank'].values}]

    dcl_path = create_dclut(lfp_path, [-1, chan_num], dcl_path=dcl_path, 
                            dtype='int16', data_name='data', data_unit='au', 
                            scales = scales)
    return dcl_path

def calc_lfp(raw_data, fs, lfp_cutoff=200, down_factor=30, offset=0,
             ignore_chans=[]):
    """
    Calculate the LFP from raw data.
    
    Parameters
    ----------
    raw_data : np.ndarray
        Raw data with time as the first axis and channels as the second axis.
    fs : numeric
        Sampling frequency of the raw data.
    lfp_cutoff : numeric, optional
        Cutoff frequency for low-pass filter in Hz. Default is 200 Hz.
    down_factor : numeric, optional
        Factor for downsampling the LFP data, i.e. number of skipped
        samples. Default is 30.
    offset : numeric, optional
        Offset for downsampling the LFP data. Default is 0.
    ignore_chans : list of int, optional
        List of channel indices to exclude from LFP filtering. 
        Default is [].
        
    Returns
    -------
    lfp_data : np.ndarray
        LFP data.
    """

    # determine channels to process
    chan_num = raw_data.shape[1]
    chans = np.where(np.isin(np.arange(chan_num), ignore_chans, invert=True))[0]

    # create butterworth filter
    b, a = butter(2, lfp_cutoff/(fs/2), 'low')

    # initialize LFP data
    lfp_data = raw_data.copy()
    
    # filter the data
    lfp_data[:, chans] = filtfilt(b, a, raw_data[:, chans], axis=0)

    # downsample the data
    lfp_data = lfp_data[offset::down_factor]

    return lfp_data
