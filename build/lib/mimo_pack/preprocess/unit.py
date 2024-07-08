# Unit preprocessing related functions
# Author: Drew Headley
# Created: 2024-06-11

import os
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from ..fileio.spikeglx import read_meta, get_meta_path

# Spike waveform generation
def sample_waveforms(times, bin_memmap, fs=30000, pre=1, post=2, sample_max=1000, sy_chan=384):
    """
    Load spike waveforms from binary file.
    
    Parameters
    ----------
    times : np.ndarray
        Spike times in seconds.
    bin_memmap : np.memmap
        Numpy memory mapped binary file.
    pre : numeric, optional
        Time before spike in ms. Default is 1 ms.
    post : numeric, optional
        Time after spike in ms. Default is 2 ms.
    sample_max : numeric, optional
        Maximum number of spikes to sample. Default is 1000.
    sy_chan : int, optional
        Channel number for sync signal. If None, no sync channel present.
        Default is 384.

    Returns
    -------
    waveform : np.ndarray
        Spike waveforms with shape (n_spikes, n_samples, n_channels).
    sub_flag : bool
        Flag indicating if fewer than sample_max spikes were sampled.
    """

    # convert pre and post durations to samples
    pre_samp = int(pre*fs/1000)
    post_samp = int(post*fs/1000)
    chan_num = bin_memmap.shape[1]

    # convert spike times to indices
    inds = (times*fs).astype(np.int64)
    sub_flag = True
    if inds.size > sample_max:
        inds = np.sort(np.random.choice(inds, 1000))
        sub_flag = False

    waveforms = np.zeros((inds.size, pre_samp+post_samp, chan_num))
    for i, spk in enumerate(inds):
        waveforms[i] = bin_memmap[(spk-pre_samp):(spk+post_samp), :]

    # remove sy channel if present
    if sy_chan is not None:
        waveforms = np.delete(waveforms, sy_chan, axis=2)

    return waveforms, sub_flag

def mean_waveform(times, bin_memmap, **kwargs):
    """
    Calculate the mean waveform of the spike.

    Parameters
    ----------
    times : np.ndarray
        Spike times in seconds.
    bin_memmap : np.memmap
        Numpy memory mapped binary file.
    kwargs : dict
        Keyword arguments for sample_waveforms.
    
    Returns
    -------
    mean_waveform : np.ndarray
        Mean unit waveform with shape (n_samples, n_channels)
    sub_flag : bool
        Flag indicating if fewer than sample_max spikes were sampled.
    """

    waveforms, sub_flag = sample_waveforms(times, bin_memmap, **kwargs)
    waveform = waveforms-np.mean(waveforms, axis=1)[:, np.newaxis, :]
    mean_waveform = np.mean(waveform, axis=0)
    return mean_waveform, sub_flag


def waveform_peak(waveform, scale=1):
    """
    Get the properties of the unit waveform peak
    
    Parameters
    ----------
    waveform : np.ndarray
        Unit waveform with shape (n_samples, )
    scale : numeric, optional
        Scale factor for the waveform to convert to voltage. Default is 1.
        
    Returns
    -------
    loc : int
        Location of the peak in samples
    amp : numeric
        Amplitude of the peak
    """

    loc = np.argmax(np.abs(waveform))
    amp = waveform[loc]*scale

    return loc, amp

def waveform_halfwidth(waveform, fs=30000):
    """
    Get the width of the unit waveform at half maximum
    
    Parameters
    ----------
    waveform : np.ndarray
        Unit waveform with shape (n_samples, )
    fs : numeric, optional
        Sampling frequency of the waveform in Hz. Default is 30000 Hz.
        
    Returns
    -------
    width : numeric
        Width of the waveform at half maximum in ms
    """

    up_factor = 10
    # upsample waveform by a factor of 10 with interpolation
    waveform = np.interp(np.linspace(0, waveform.size-1, waveform.size*up_factor), 
                         np.arange(waveform.size), waveform)
    loc, amp = waveform_peak(waveform)

    half_max = amp/2

    # count indices above half max starting from peak
    if amp > 0:
        left_side = np.where(waveform[loc:0:-1] < half_max)[0][0]
        right_side = np.where(waveform[loc:] < half_max)[0][0]
    else:
        left_side = np.where(waveform[loc:0:-1] > half_max)[0][0]
        right_side = np.where(waveform[loc:] > half_max)[0][0]
    
    width = (left_side + right_side - 1)/((fs*up_factor)/1000)

    return width


def classify_unit(frate, halfwidth, region='CTX'):
    """
    Classifies a unit based on waveform properties

    Parameters
    ----------
    frate : numeric
        Firing rate of the unit in Hz
    halfwidth : numeric
        Width of the waveform at half maximum in ms
    region : str, optional
        Brain region of the unit. Default is 'CTX', cortex.

    Returns
    -------
    uclass : str
        Classification of the unit. For cortex, the classes are:
        'RS' - Regular spiking
        'FS' - Fast spiking
        'UN' - Unidentified
    """

    if region == 'CTX':
        if (frate > 2) and (halfwidth < 0.15):
            uclass = 'FS'
        elif (frate < 10) and (halfwidth > 0.15):
            uclass = 'RS'
        else:
            uclass = 'UN'
    
    return uclass


