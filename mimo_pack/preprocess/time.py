# Timing preprocessing related functions
# Author: Drew Headley
# Created: 2024-07-08

import numpy as np
from tqdm import tqdm
from scipy.stats import zscore
from scipy.ndimage import binary_opening

def align_sync(sync_t, sync_r, time_r, matchlen=30):
    """
    Aligns two synchronization signals based on shared pulse durations.

    Parameters
    ----------
    sync_o : np.ndarray
        Synchronization signal from the target recording
    sync_r : np.ndarray
        Synchronization signal from the reference recording
    time_r : np.ndarray
        Time points corresponding to the reference recording
    matchlen : int
        Length of the pulse to match in samples

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
    pulse_pairs = align_series(dur_r, dur_t, matchlen)
    print(pulse_pairs)
    align_t = edges_t[pulse_pairs[:,1],0]
    align_r = edges_r[pulse_pairs[:,0],0]

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

def align_series(ser1, ser2, matchlen, verbose=True):
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