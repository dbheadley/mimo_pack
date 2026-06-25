"""Load Phy format spikes
Author: Drew B. Headley
"""

import pynapple as nap
import pandas as pd
import numpy as np
import dclut as dcl
import xarray as xr
from tqdm import tqdm
import sys
import os

from mimo_pack.fileio.ap import load_ap_windows_raw

def load_phy_spikes(phy_dir, dcl_file=None, suffix="", cluster_ids=None, verbose=True):
    """
    Step 1: Loads spike times and IDs from Phy files into a pynapple TsGroup.
    Returns Tsd objects where the data payload is the raw sample index.
    """
    if verbose: print("Loading phy files...")
    
    fpaths = {
        "par": os.path.join(phy_dir, "params.py"),
        "spk": os.path.join(phy_dir, f"spike_times{suffix}.npy"),
        "clu": os.path.join(phy_dir, f"spike_clusters{suffix}.npy")
    }

    # Verify essential files
    for fpath in fpaths.values():
        if not os.path.isfile(fpath):
            raise RuntimeError(f"Cannot find {fpath}")

    # Load parameters
    sys.path.append(phy_dir)
    import params
    samp_rate = params.sample_rate

    # Load spike indices and clusters
    spk_times = np.load(fpaths["spk"])
    clu_ids = np.load(fpaths["clu"])

    if cluster_ids is None:
        clu_id_list = np.unique(np.sort(clu_ids))
    else:
        clu_id_list = cluster_ids

    # Time Synchronization via DCL
    explicit_times = False
    if dcl_file is not None:
        if os.path.isfile(dcl_file):
            if verbose: print("Syncing times with dclut file...")
            spks_dcl = dcl.dclut(dcl_file)
            time_arr = spks_dcl.scale_values('time')
            explicit_times = True
            
            # Find session boundaries
            if np.isnan(time_arr[0]):
                start_ind = np.where(np.diff(np.isnan(time_arr).astype(int))<0)[0][0]+1
            else:
                start_ind = 0
            
            if np.isnan(time_arr[-1]):
                end_ind = np.where(np.diff(np.isnan(time_arr).astype(int))>0)[0][0]
            else:
                end_ind = time_arr.size - 1

            sess_start = time_arr[start_ind]
            sess_end = time_arr[end_ind]
        else:
            raise RuntimeError(f"Cannot find {dcl_file}")
    else:
        sess_start = 0
        sess_end = spk_times.max() / samp_rate
    
    sess_set = nap.IntervalSet(sess_start, sess_end)

    # Create Tsd objects (Time Series Data)
    # We store the raw indices (curr_inds) as the data payload
    spk_dict = {}
    iter_list = tqdm(clu_id_list, desc="Loading Spikes") if verbose else clu_id_list

    for cid in iter_list:
        mask = clu_ids == cid
        curr_inds = np.sort(spk_times[mask])
        
        if explicit_times:
            # Filter by session bounds
            curr_inds = curr_inds[(curr_inds > start_ind) & (curr_inds < end_ind)]
            curr_times = time_arr[curr_inds]
            curr_times = curr_times[~np.isnan(curr_times)]
        else:
            curr_times = curr_inds / samp_rate

        # Store indices in 'd' argument
        spk_dict[cid] = nap.Tsd(t=curr_times, d=curr_inds, time_units="s", time_support=sess_set)

    spks = nap.TsGroup(spk_dict)

    # Load Cluster Groups/Labels
    group_fpath = os.path.join(phy_dir, f"cluster_group{suffix}.tsv")
    if os.path.isfile(group_fpath):
        if verbose: print("Loading cluster classes...")
        clu_group = pd.read_csv(group_fpath, sep="\t", index_col="cluster_id")
        clu_group = clu_group.rename(columns={'SASLabel': 'class'})
        clu_group = clu_group.sort_index()
        spks.set_info(clu_group.loc[clu_id_list])

    return spks

def as_pynapple(phy_dir, dcl_file=None, suffix="", cluster_ids=None,
                verbose=True):
    """
    Loads spike times from Phy formatted files as a pynapple time series group
    
    Parameters
    ----------
    phy_dir : string
        Path to the directory holding the Phy files
    dcl_file : string
        Full file path to a dclut json file with a 'time', 'ch_x', 'ch_y' and 'ch_shank' scales. 
        Used when just dividing time index by sample rate will not suffice and 
        explicit time points are required because multiple files had to be 
        synchronized with each other. The dclut file will also be used to determine
        the mean spike waveform and spike peak location (based on 'ch_x' and 'ch_y').
    suffix : string
        Suffix to add to the end of file names when loading files.
    cluster_ids : list of int, optional
        List of cluster IDs to load. If None, all clusters will be loaded.
    verbose : bool, optional
        Whether to print progress messages. Default is True.
    
    Returns
    ----------
    spks : pynapple TsGroup
        A pynapple group of time series objects
    """

    # file paths that are necessary
    fpaths = {}
    fpaths["par"] = os.path.join(phy_dir, "params.py")
    fpaths["spk"] = os.path.join(phy_dir, "spike_times{}.npy".format(suffix))
    fpaths["clu"] = os.path.join(phy_dir, "spike_clusters{}.npy".format(suffix))

    # check that essential files are present
    for fpath in fpaths.values():
        if not os.path.isfile(fpath):
            raise RuntimeError("Cannot find {}".format(fpath))

    # parameters needed for converting spike indices to times
    if verbose:
        print("Loading phy files")
    sys.path.append(phy_dir)
    import params as params

    samp_rate = params.sample_rate

    # load spike indices and their cluster IDs
    spk_times = np.load(fpaths["spk"])
    clu_ids = np.load(fpaths["clu"])

    if cluster_ids is None:
        clu_id_list = np.unique(np.sort(clu_ids))
    else:
        clu_id_list = cluster_ids

    # if dclut file is present, use it to establish timing
    explicit_times = False
    if dcl_file is not None:
        if os.path.isfile(dcl_file):
            if verbose:
                print("Loading times from dclut file")
            spks_dcl = dcl.dclut(dcl_file)
            time_arr = spks_dcl.scale_values('time')
            explicit_times = True
        else:
            raise RuntimeError("Cannot find {}".format(dcl_file))

    # define the beginning and end of the session
    if explicit_times:
        # find first entry in time_arr that is nan
        if np.isnan(time_arr[0]):
            start_ind = np.where(np.diff(np.isnan(time_arr).astype(int))<0)[0][0]+1
        else:
            start_ind = 0
        
        # find last entry in time_arr that is nan
        if np.isnan(time_arr[-1]):
            end_ind = np.where(np.diff(np.isnan(time_arr).astype(int))>0)[0][0]
        else:
            end_ind = time_arr.size - 1

        sess_start = time_arr[start_ind]
        sess_end = time_arr[end_ind]
    else:
        sess_start = 0
        sess_end = spk_times.max() / samp_rate
    sess_set = nap.IntervalSet(sess_start, sess_end)

    # assign spikes to clusters and create time series group
    spk_dict = {}
    spk_inds_dict = {}
    if verbose:
        print("Assigning spikes to clusters")
        clu_iter = tqdm(clu_id_list, desc="Assigning: ", unit="cluster")
    else:
        clu_iter = clu_id_list

    for id in clu_iter:
        curr_spk_inds = np.sort(spk_times[clu_ids == id])
        if explicit_times:    
            # remove spikes with indices outside of the session
            curr_spk_inds = curr_spk_inds[curr_spk_inds > start_ind]
            curr_spk_inds = curr_spk_inds[curr_spk_inds < end_ind]

            # convert spike indices to times
            curr_spk_times = time_arr[curr_spk_inds]

            # remove spike times that are undefined (nan)
            curr_spk_times = curr_spk_times[~np.isnan(curr_spk_times)]

        else:
            curr_spk_times = spk_times[clu_ids == id] / samp_rate

        spk_inds_dict[id] = curr_spk_inds # for waveform extraction
        spk_dict[id] = nap.Ts(curr_spk_times, time_units="s", time_support=sess_set)
    spks = nap.TsGroup(spk_dict)

    # add cluster class to the spike group
    if verbose:
        print("Loading cluster classes")
    group_fpath = os.path.join(phy_dir, "cluster_group{}.tsv".format(suffix))
    if os.path.isfile(group_fpath):
        clu_group = pd.read_csv(group_fpath, sep="\t", index_col="cluster_id")
        clu_group = clu_group.rename(columns={'SASLabel': 'class'})
        clu_group = clu_group.sort_index()

        # keep only clusters that were loaded
        clu_group = clu_group.loc[clu_id_list]
        
        spks.set_info(clu_group)
    
    # if dcl_file provided, get spike waveform properties:
    # mean waveform on strongest channels
    # peak voltage dist across channels
    # indices of peak channels
    # peak channel coordinates
    if dcl_file is not None:
        if verbose:
            print("Getting spike waveform properties")
            clu_iter = tqdm(clu_id_list, desc="Assigning: ", unit="cluster")
        else:
            clu_iter = clu_id_list

        wave_list = []
        samp_num = 1000
        ind_max = spks_dcl.dcl['file']['shape'][0]
        wave_win = np.array([[-30], [60]])
        x_pos = spks_dcl.scale_values(scale='ch_x')
        y_pos = spks_dcl.scale_values(scale='ch_y')
        shank = spks_dcl.scale_values(scale='ch_shank')
        for id in clu_iter:
            spks_dcl.reset()

            spk_inds = spk_inds_dict[id]
            spk_inds = spk_inds[spk_inds < (ind_max-60)]
            spk_inds = spk_inds[spk_inds > 30]
            num_spks = spk_inds.size

            if num_spks > samp_num:
                spk_inds = np.sort(np.random.choice(spk_inds, samp_num))

            # get spike waveforms
            bin_file = os.path.join(os.path.dirname(dcl_file), spks_dcl.dcl['file']['name'])
            waves = load_ap_windows_raw(bin_file, spk_inds, 30, 60, n_channels=spks_dcl.dcl['file']['shape'][1])

            # get mean spike waveform
            # subtract trend baseline from each spike
            waves = waves - np.linspace(waves[0,:,:], waves[-1,:,:], 90) 
            mean_wave = np.mean(waves, axis=2)

            # identify 8 channels near where the spike waveform is largest
            # get min or max (whicher is larger magnitude) for each channel
            wave_amp = np.linalg.norm(mean_wave, axis=0)
            peak_ind = np.argsort(wave_amp)[-1]
            peak_dists = (x_pos - x_pos[peak_ind])**2 + (y_pos - y_pos[peak_ind])**2
            near_inds = np.argsort(peak_dists)[:8]
            
            # sort near_inds by amplitude of the waveform
            near_inds = near_inds[np.argsort(wave_amp[near_inds])] 
            waveform = mean_wave[:, near_inds]
         
            x_near = x_pos[near_inds]
            y_near = y_pos[near_inds]
            shank_near = shank[near_inds]
            
            # sort channels by y then x position
            sort_inds = np.lexsort((x_pos, y_pos, shank))
            
            # create xarray for full waveform
            full_waveform_xr = xr.DataArray(
                mean_wave[:, sort_inds],
                dims=('time', 'channel'),
                coords={
                    'time': np.arange(-30, 60),
                    'channel': np.arange(mean_wave.shape[1]),
                    'ch_x': ('channel', x_pos[sort_inds]),
                    'ch_y': ('channel', y_pos[sort_inds]),
                    'ch_shank': ('channel', shank[sort_inds])
                }
            )

            wave_list.append({'waveform': waveform, 'inds': near_inds, 
                              'x': x_near, 'y': y_near, 'shank':shank_near, 'full_waveform': full_waveform_xr})
        spks.set_info(x=[w['x'][-1] for w in wave_list])
        spks.set_info(y=[w['y'][-1] for w in wave_list])
        spks.set_info(shank=[w['shank'][-1] for w in wave_list])
        spks.set_info(waveform=wave_list)

    return spks


# Debug test
if __name__ == "__main__":
    test_dir = "../../TestData/phy/"
    test_spks = as_pynapple(test_dir)
    print(test_spks)
