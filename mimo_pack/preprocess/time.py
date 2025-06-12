# Timing preprocessing related functions
# Author: Drew Headley
# Created: 2024-07-08

import numpy as np
import pandas as pd
import pdb
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

def align_sync_barcode(bc_t, bc_r, time_r, matchlen=10, **kwargs):
    """
    Aligns two synchronization signals based on barcodes.

    Parameters
    ----------
    bc_t : np.ndarray
        Barcode sequence from the target recording
    bc_r : np.ndarray
        Barcode sequence from the reference recording
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

    # Align barcoe sequences
    bc_pairs = align_sequence_exact(bc_t, bc_r, matchlen, **kwargs)

    # Match target barcodes to reference time points
    time_t = time_r[bc_pairs[:,1]]
    align_table = np.stack([bc_pairs[:,0], time_t], axis=1)

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
    Align two series based on similarity of their sequential values.

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

def align_sequence_exact(ser1, ser2, matchlen, verbose=False):
    """
    Align two series based on exact matches of their sequential values.

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

    ser1 = ser1.ravel()
    ser2 = ser2.ravel()
    start2 = 0
    left2 = 0
    s1 = 0
    s2 = 0
    pairs = []

    if verbose:
        pbar = tqdm(total=len(ser1), desc='Aligning barcodes', unit='frames')

    while s1 < len(ser1): # step through target series
        run_counter = 0
        if verbose:
            pbar.n = s1
            pbar.refresh()
        while s2 < len(ser2): # step through reference series
            if ser1[s1+run_counter] == ser2[s2]: # if values match, count matches
                if run_counter == 0:
                    start2 = s2
                if (s1+run_counter+1) == len(ser1): # if end of target, save match
                    if run_counter >= matchlen:
                        pairs.append([np.arange(s1,len(ser1)), 
                                        np.arange(start2,s2+1)])
                    s1 += run_counter
                    break
                run_counter += 1
            else: # when values stop matching, check if run is long enough and save
                if run_counter >= matchlen:
                    pairs.append([np.arange(s1,s1+run_counter), 
                                np.arange(start2,s2)])
                    left2 = s2
                    s1 += run_counter
                    run_counter = 0             
                    break
                run_counter = 0
            s2 += 1
        s2 = left2
        s1 += 1

    pairs = np.concatenate([np.stack(p,axis=1) for p in pairs])
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

def align_sync_video_barcode(table_path, path_r, barcode_name, sync_r, sync_scale_name, verbose=False):
    """
    Align video barcode table to a binary with dclut meta with serialized sync pulses.
    
    Parameters
    ----------
    table_path : str
        File path to barcode table
    path_r : str
        File path to dclut object for the reference recording
    barcode_name : str
        Name of the barcode column in the table
    sync_r : dict or list of dicts
        Dictionary specifying the sync index for the reference recording

    Optional
    --------
    verbose : bool
        Print progress of alignment

    Returns
    -------
    path_t : csv path
        File path to csv file for the target recording
    """

    # data loaded in chunks to avoid memory issues
    # load the video barcode table
    
    if verbose:
        print('Reading video target file')
    vid_df = pd.read_csv(table_path, index_col=0)
    
    # load the reference serialized barcode signal
    if verbose:
        print('Reading recording reference file')
    dcl_r = dclut(path_r, verbose=verbose)
    max_sec = dcl_r.scale_values(sync_scale_name)[-1]
    min_edges = np.append(np.arange(0, max_sec, 60), max_sec)
    min_intervals = np.stack([min_edges[:-1], min_edges[1:]],axis=1)
    dcl_r.intervals({sync_scale_name: min_intervals}, select_mode='split')
    dcl_r.points(sync_r)
    sync = dcl_r.read(format='xarray')
    sync_r = xr.concat(sync, dim=sync_scale_name)
    
    # deserialize the barcode signal
    if verbose:
        print(f'Barcode detection')
    ser_trace = sync_r.values
    ser_times = sync_r[sync_scale_name].values
    ser_df, bc_detect = deserialize_barcode(ser_trace, ser_times)

    if verbose:
        print(f'Barcodes detected: {bc_detect:.2f}%')

    # align the two recordings
    align_table = align_sync_barcode(vid_df[barcode_name].values, 
                                     ser_df['barcode'].values, 
                                     ser_df['timestamp'].values, 
                                     matchlen=10, verbose=verbose)
    
    # format alignment table
    align_df = pd.DataFrame(align_table, columns=['frame', 'time'])
    align_df['frame'] = align_df['frame'].astype(int)
    align_df = align_df.set_index('frame')


    # add times from alignment table to the video table
    vid_df = vid_df.join(align_df)
    vid_df.to_csv(table_path, na_rep='', index=True, index_label='frame')

    # calculate percentage of aligned frames
    aligned = np.sum(~np.isnan(vid_df['time'])) / len(vid_df) * 100
    if verbose:
        print(f'Aligned frames: {aligned:.2f}%')

    return table_path


def deserialize_barcode(ser_sig, ser_time):
    """
    Deserializes a binary barcode signal into a sequence of barcodes
    with their corresponding timestamps.

    Parameters
    ----------
    ser_sig : np.ndarray
        Binary array representing the serialized barcode signal.
    ser_time : np.ndarray
        Array of timestamps corresponding to the serialized signal.
    verbose : bool, optional
        If True, prints progress messages. Default is False.

    Returns
    -------
    bc_df : DataFrame
        DataFrame containing the deserialized barcodes and their timestamps.
    bc_detected : numeric
        Percentage of barcodes detected from the serialized signal.
    """

    # Ensure trace is binary
    ser_sig = (np.bitwise_and(ser_sig, 2**6)>0).astype(int)

    # identify the beginning and end of each pulse
    onsets = np.where(np.diff(ser_sig,axis=0) > 0)[0] + 1
    offsets = np.where(np.diff(ser_sig,axis=0) < 0)[0] + 1

    # Trim incomplete bits
    if offsets[0] < onsets[0]:
        offsets = offsets[1:]
    if onsets[-1] > offsets[-1]:
        onsets = onsets[:-1]

    # create array of pulses
    pulses = np.stack([onsets, offsets], axis=1)
    pulse_durs = np.diff(pulses, axis=1)
    bit_len = np.percentile(pulse_durs, 5)
    half_len = bit_len // 2

    # Create sequence of discrete bits
    bit_count = 0
    bit_seq = np.zeros(int(np.ceil(pulses[-1,0] / bit_len)), dtype=bool)
    ind_seq = np.zeros(bit_seq.shape, dtype=int)

    for i in range(len(onsets)-1):
        high_num = int(np.round(pulse_durs[i] / bit_len))
        bit_seq[bit_count:(bit_count+high_num)] = True
        ind_seq[bit_count:(bit_count+high_num)] = pulses[i,0] + half_len + np.arange(high_num) * bit_len
        bit_count += high_num

        low_num = int(np.round((pulses[i+1, 0] - pulses[i,1]) / bit_len))
        ind_seq[bit_count:(bit_count+low_num)] = pulses[i,1] + half_len + np.arange(low_num) * bit_len
        bit_count += low_num

    # Identify barcodes
    bc_seq = []
    bc_len = 10 # Number of bits in each barcode
    pos_vals = np.pow(2, np.arange(bc_len))[:, np.newaxis]
    bc_chnk_len = 199  # Odd number for how many barcodes to look forward when confirming offset
    i = 0
    offset = 0

    pbar = tqdm(total=len(bit_seq), desc='Deserializing barcodes', unit='bits')
    while i < (len(bit_seq)-(bc_len*(bc_chnk_len+2))):
        pbar.n = i
        pbar.refresh()
        # Set offset by identifying the 'constant' pulse and clock pulse
        bc_chnk_sync = bit_seq[i+offset+(bc_len*np.arange(bc_chnk_len))]
        if not np.all(bc_chnk_sync):
            offset += 1
            if offset > (bc_len-1):
                offset = 0
                i += 1
            continue

        # Extract chunks when valid offset is found, compute barcode value
        curr_chunk = bit_seq[(i+offset):(i+offset+(bc_len*bc_chnk_len))].reshape(bc_chnk_len, bc_len)
        ind_chunk = ind_seq[(i+offset):(i+offset+(bc_len*bc_chnk_len)):bc_len]
        bc_seq.append(np.column_stack([ind_chunk, np.dot(curr_chunk,pos_vals)]))
        i += (bc_len * bc_chnk_len)

    bc_seq = np.vstack(bc_seq)

    # Display percentage of data that could be decoded
    num_bcs = bc_seq.shape[0]
    bc_detected = num_bcs / (len(bit_seq) / bc_len) * 100

    # Add timestamps to barcodes
    #bc_seq = np.column_stack([bc_seq, ser_time[bc_seq[:, 0]]])
    bc_df = pd.DataFrame({"index": bc_seq[:,0].astype(int), 
                          "barcode": bc_seq[:,1].astype(int), 
                          "timestamp": ser_time[bc_seq[:,0]]})

    return bc_df, bc_detected