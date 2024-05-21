# LFP preprocessing related functions
# Author: Drew Headley
# Created: 2024-05-19

import os
import sys
sys.path.append('../Code/')
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, filtfilt

from fileio.spikeglx import read_meta, get_meta_path

def make_lfp_file_spikeglx(bin_path, lfp_cutoff=500, lfp_fs=1000, suffix='lfp'):
    """
    Makes LFP file from raw SpikeGLX binary. Overwrites existing LFP file.

    Parameters
    ----------
    bin_path : str
        Path to binary file.
    lfp_cutoff : numeric, optional
        Cutoff frequency for low-pass filter in Hz. Default is 500 Hz.
    lfp_fs : numeric, optional
        Sampling frequency of LFP data in Hz. Default is 1000 Hz.
    suffix : str, optional
        Suffix for LFP file. Default is 'lfp'.

    Returns
    -------
    lfp_path : str
        Path to LFP file.
    """

    # Check if the binary file exists
    if not os.path.exists(bin_path):
        raise FileNotFoundError('Binary file {} not found'.format(bin_path))
    
    # Get the meta data
    meta_path = get_meta_path(bin_path)
    meta = read_meta(meta_path)

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

    # iterate through chunks
    for i in tqdm(range(len(chunks)-1)):
        # read in chunk of data
        binf.seek(chunks[i]*step_bytes)
        chunk_len = chunks[i+1]-chunks[i]
        bin_data = np.fromfile(binf, dtype='int16', count=chunk_len*chan_num)
        bin_data = bin_data.reshape((chunk_len, chan_num))
        lfp_data = bin_data.copy()

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


def calc_lfp(raw_data, fs, lfp_cutoff=500, down_factor=30, offset=0,
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
        Cutoff frequency for low-pass filter in Hz. Default is 500 Hz.
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
