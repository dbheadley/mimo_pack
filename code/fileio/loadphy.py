"""Load Phy format spikes
Author: Drew B. Headley
"""

import pynapple as nap
import pandas as pd
import numpy as np
import sys
import os


def as_pynapple(phy_dir, time_file=None):
    """
    Loads spike times from Phy formatted files as a pynapple time series group
    
    Parameters
    ----------
    phy_dir : string
        path to the directory holding the Phy files
    time_file : string
        full file path to a binary file of time steps. Assumes times are stored
        as a flat array of doubles. Used when just dividing time index by sample
        rate will not suffice and explicit time points are required because 
        multiple files had to be synchronized with each other.
    
    Returns
    ----------
    spks : pynapple TsGroup
        A pynapple group of time series objects
    """

    # file paths that are necessary
    fpaths = {}
    fpaths["par"] = os.path.join(phy_dir, "params.py")
    fpaths["spk"] = os.path.join(phy_dir, "spike_times.npy")
    fpaths["clu"] = os.path.join(phy_dir, "spike_clusters.npy")

    # check that essential files are present
    for fpath in fpaths.values():
        if not os.path.isfile(fpath):
            raise RuntimeError("Cannot find {}".format(fpath))

    # parameters needed for converting spike indices to times
    sys.path.append(phy_dir)
    import params as params

    samp_num = params.n_samples_dat
    samp_rate = params.sample_rate

    # load spike indices and their cluster IDs
    spk_times = np.load(fpaths["spk"])
    clu_ids = np.load(fpaths["clu"])
    clu_id_list = np.unique(np.sort(clu_ids))

    # if time dat file is present, use it to establish timing
    explicit_times = False
    if time_file is not None:
        if os.path.isfile(time_file):
            time_list = np.fromfile(time_file, dtype="double", count=-1)
            explicit_times = True
        else:
            raise RuntimeError("Cannot find {}".format(time_file))

    # define the beginning and end of the session
    if explicit_times:
        sess_start = time_list[0]
        sess_end = time_list[-1]
    else:
        sess_start = 0
        sess_end = samp_num / samp_rate
    sess_set = nap.IntervalSet(sess_start, sess_end)

    # assign spikes to clusters and create time series group
    spk_dict = {}
    for id in clu_id_list:
        if explicit_times:
            curr_spk_times = time_list[spk_times[clu_ids == id]]
        else:
            curr_spk_times = spk_times[clu_ids == id] / samp_rate
        spk_dict[id] = nap.Ts(curr_spk_times, time_units="s", time_support=sess_set)
    spks = nap.TsGroup(spk_dict)

    # if a cluster properties file is present, add the cluster properties
    # to the spike group
    prop_fpath = os.path.join(phy_dir, "cluster_props.tsv")
    if os.path.isfile(prop_fpath):
        fpaths["pos"] = os.path.join(phy_dir, "channel_positions.npy")
        fpaths["map"] = os.path.join(phy_dir, "channel_map.npy")
        chan_map = np.load(fpaths["map"])
        chan_pos = np.load(fpaths["pos"])
        clu_props = pd.read_csv(prop_fpath, sep="\t", index_col="cluster_id")
        clu_props["x_pos"] = clu_props["peak_chan"].map(
            lambda x: chan_pos[np.flatnonzero(chan_map == x), 0].item()
        )
        clu_props["y_pos"] = clu_props["peak_chan"].map(
            lambda x: chan_pos[np.flatnonzero(chan_map == x), 1].item()
        )
        spks.set_info(clu_props)

    # if a templates file is present, add the spike template to the spike group
    tplt_fpath = os.path.join(phy_dir, "templates.npy")
    if os.path.isfile(tplt_fpath):
        spk_waves = np.load(tplt_fpath)
        spk_waves = [
            np.squeeze(x) for x in np.split(spk_waves, spk_waves.shape[0], axis=0)
        ]

        # full spike waveform template
        spks.set_info(full_wave=pd.Series(spk_waves, index=clu_id_list))

        # peak channel spike waveform
        peak_waves = [
            x[:, np.flatnonzero(chan_map == y)].T
            for x, y in zip(spk_waves, spks.get_info("peak_chan").values)
        ]
        spks.set_info(peak_wave=pd.Series(peak_waves, index=clu_id_list))

    return spks


# Debug test
if __name__ == "__main__":
    test_dir = "../../TestData/phy/"
    test_spks = as_pynapple(test_dir)
    print(test_spks)
