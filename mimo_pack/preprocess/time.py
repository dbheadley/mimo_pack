# Timing preprocessing related functions
# Author: Drew Headley
# Created: 2024-07-08

import numpy as np
from tqdm import tqdm
from scipy.stats import zscore
from scipy.ndimage import binary_opening
from dclut import dclut
import xarray as xr

def align_sync(sync_t, sync_r, time_r, matchlen=30, **kwargs):
    """
    Aligns two synchronization signals based on shared pulse durations.

    Parameters
    ----------
    sync_t : np.ndarray
        Synchronization signal from the target recording
    sync_r : np.ndarray
        Synchronization signal from the reference recording
    time_r : np.ndarray
        Time points corresponding to the reference recording
    matchlen : int
        Length of the pulse to match in samples

    Optional
    --------
    kwargs : dict
        Additional keyword arguments to pass to align_sequence

    Returns
    -------
    align_table : np.ndarray
        Nx2 array of time points from the reference recording that 
        correspond to the target recording. First column is the index
        in the target recording, second column is the corresponding time 
        point from the reference recording.
    """

    # ensure consistent format
    sync_t = sync_t.ravel()
    sync_r = sync_r.ravel()
    time_r = time_r.ravel()

    # Convert to boolean
    sync_t = np.where(zscore(sync_t) > 0, 1, 0)
    sync_r = np.where(zscore(sync_r) > 0, 1, 0)

    # Get pulse properties sequence
    edges_t = find_pulse_edges(sync_t)
    edges_r = find_pulse_edges(sync_r)

    # Calculate pulse durations
    dur_t = np.diff(edges_t, axis=1)
    dur_r = np.diff(edges_r, axis=1)

    # Align pulse sequences
    pulse_pairs = align_sequence(dur_t, dur_r, matchlen, **kwargs)

    align_t = edges_t[pulse_pairs[:,0],0]
    align_r = edges_r[pulse_pairs[:,1],0]

    # Match pulse edges to time points
    time_edge_r = time_r[align_r]

    align_table = np.stack([align_t, time_edge_r], axis=1)

    return align_table


def find_pulse_edges(pulse_ser):
    """
    Find pulses return their start and end indices

    Parameters
    ----------
    pulse_ser : np.array
        Binary array of pulses
    
    Returns
    -------
    edges : np.ndarray
        Nx2 array of pulse start and end indices.
        First column is start, second column is end.
    """

    # prepare pulse series for edge detection, enable edge
    # detection at the beginning and end of the series
    pulse_ser = np.pad(pulse_ser, (1,1), 'constant')

    # find pulse starts and ends
    starts = np.where(np.diff(pulse_ser) == 1)[0]
    ends = np.where(np.diff(pulse_ser) == -1)[0]-1

    # stack starts and ends into Nx2 array
    edges = np.stack([starts, ends], axis=1)
    
    return edges

def align_sequence(ser1, ser2, matchlen, verbose=False):
    """
    Align two series based on similarity of their value sequential values.

    Parameters
    ----------
    ser1 : np.ndarray
        First series to align
    ser2 : np.ndarray
        Second series to align
    matchlen : int
        Length of the matching window

    Optional
    --------
    verbose : bool
        Print progress of alignment

    Returns
    -------
    pairs : np.ndarray
        Nx2 array of indices in the two series that are aligned
    """

    if matchlen % 2 == 1:
        raise ValueError('Match Length must be an even number')

    ser1 = ser1.ravel()
    ser2 = ser2.ravel()

    ser1 = np.pad(zscore(ser1), (matchlen, matchlen), constant_values=np.nan)
    ser2 = np.pad(zscore(ser2), (matchlen, matchlen), constant_values=np.nan)
   

    stop1 = len(ser1) - matchlen
    stop2 = len(ser2) - matchlen
    s2 = matchlen + 1
    pairs = []
    if verbose:
        outer_iter = tqdm(range(matchlen+1, stop1))
    else:
        outer_iter = range(matchlen+1, stop1)

    for s1 in outer_iter:
        for s2off in range(0,stop2-s2):
            serdiff = np.abs(ser1[(s1-matchlen):(s1+matchlen)]
                             -ser2[(s2+s2off-matchlen):(s2+s2off+matchlen)])
            serdiff = binary_opening(serdiff<1, np.ones(matchlen))
            if serdiff[matchlen]:
                s2 += s2off
                pairs.append([s1, s2])
                break

    pairs = np.array(pairs) - matchlen - 1
    return pairs

def align_sync_dclut(path_t, path_r, sync_t, sync_r, sync_scale_name, verbose=False):
    """
    Align two binary files with dclut meta based on sync pulses.
    
    Parameters
    ----------
    path_t : dclut path
        File path to dclut object for the target recording
    path_r : dclut path
        File path to dclut object for the reference recording
    sync_t : dict or list of dicts
        Dictionary specifying the sync index for the target recording
    sync_r : dict or list of dicts
        Dictionary specifying the sync index for the reference recording

    Optional
    --------
    verbose : bool
        Print progress of alignment

    Returns
    -------
    path_t : dclut path
        File path to dclut object for the target recording
    """

    # data loaded in chunks to avoid memory issues
    # load the target recording
    if verbose:
        print('Reading target file')
    dcl_t = dclut(path_t, verbose=verbose)
    # create an array of edges for 1 minute intervals
    max_sec = dcl_t.scale_values(sync_scale_name)[-1]
    min_edges = np.append(np.arange(0, max_sec, 60), max_sec)
    min_intervals = np.stack([min_edges[:-1], min_edges[1:]],axis=1)
    # configure the dclut object to read a single channel
    dcl_t.intervals({sync_scale_name: min_intervals}, select_mode='split')
    dcl_t.points(sync_t)
    sync = dcl_t.read(format='xarray')
    # concatenate the data along the time dimension using xarray
    sync_t = xr.concat(sync, dim=sync_scale_name)

    # repeat the process for the reference recording
    if verbose:
        print('Reading reference file')
    dcl_r = dclut(path_r, verbose=verbose)
    max_sec = dcl_r.scale_values(sync_scale_name)[-1]
    min_edges = np.append(np.arange(0, max_sec, 60), max_sec)
    min_intervals = np.stack([min_edges[:-1], min_edges[1:]],axis=1)
    dcl_r.intervals({sync_scale_name: min_intervals}, select_mode='split')
    dcl_r.points(sync_r)
    sync = dcl_r.read(format='xarray')
    sync_r = xr.concat(sync, dim=sync_scale_name)

    # align the two recordings
    if verbose:
        print('Aligning sync signals')
    s_r = sync_r.data
    s_t = sync_t.data
    time_r = sync_r['time'].data
    align_table = align_sync(s_t, s_r, time_r, matchlen=30, verbose=verbose)

    # if alignment is not perfect, nan the edges of the recording
    if align_table[0,0] != 0:
        align_table = np.vstack([[0,np.nan], align_table])
    
    if align_table[-1,0] != len(s_t):
        align_table = np.vstack([align_table, [len(s_t), np.nan]])

    # update the dclut object with the new alignment
    new_scale = dcl_t.dcl['scales'][sync_scale_name]
    new_scale['type'] = 'table'
    new_scale['values'] = align_table
    dcl_t.save()

    return path_t