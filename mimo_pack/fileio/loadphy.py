"""Load Phy format spikes
Author: Drew B. Headley
"""

import pynapple as nap
import pandas as pd
import numpy as np
import dclut as dcl
from tqdm import tqdm
import sys
import os

def as_pynapple(phy_dir, dcl_file=None, suffix="", cluster_ids=None, qc=False,
                verbose=True):
    """
    Loads spike times from Phy formatted files as a pynapple time series group
    
    Parameters
    ----------
    phy_dir : string
        Path to the directory holding the Phy files
    dcl_file : string
        Full file path to a dclut json file with a 'time', 'ch_x', and 'ch_y' scale. 
        Used when just dividing time index by sample rate will not suffice and 
        explicit time points are required because multiple files had to be 
        synchronized with each other. The dclut file will also be used to determine
        the mean spike waveform and spike peak location (based on 'ch_x' and 'ch_y').
    suffix : string
        Suffix to add to the end of file names when loading files.
    cluster_ids : list of int, optional
        List of cluster IDs to load. If None, all clusters will be loaded.
    qc : bool, optional
        Apply quality control to the loaded spikes. Default is False.
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
        samp_num = 100
        ind_max = spks_dcl.dcl['file']['shape'][0]
        wave_win = np.array([[-30], [60]])
        x_pos = spks_dcl.scale_values(scale='ch_x')
        y_pos = spks_dcl.scale_values(scale='ch_y')
        for id in clu_iter:
            spks_dcl.reset()

            spk_inds = spk_inds_dict[id]
            spk_inds = spk_inds[spk_inds < (ind_max-60)]
            spk_inds = spk_inds[spk_inds > 30]
            num_spks = spk_inds.size

            if num_spks > samp_num:
                spk_inds = np.sort(np.random.choice(spk_inds, samp_num))

            # get spike waveforms
            spks_dcl.intervals({'s0': (spk_inds+wave_win).T}, select_mode='split')
            waves = np.stack(spks_dcl.read(), axis=2)

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
            
            wave_list.append({'waveform': waveform, 'inds': near_inds, 
                              'x': x_near, 'y': y_near})
        spks.set_info(x=[w['x'][-1] for w in wave_list])
        spks.set_info(y=[w['y'][-1] for w in wave_list])
        spks.set_info(waveform=wave_list)

    return spks


# Debug test
if __name__ == "__main__":
    test_dir = "../../TestData/phy/"
    test_spks = as_pynapple(test_dir)
    print(test_spks)
