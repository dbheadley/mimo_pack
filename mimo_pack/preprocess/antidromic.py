# Antidromic stimulation

import numpy as np
import pandas as pd
from dclut import dclut

def get_antistim_times(dclut_path, channel):
    nidq = dclut(dclut_path)
    nidq.reset()
    nidq.points({'channel': [channel]})
    data = nidq.read(format='xarray')[0]

    max_val = data.max().values
    bin_data = data > (max_val / 2)

    times = data.time.values
    pulse_starts = times[np.where(np.diff(bin_data.values.astype(int), axis=0) == 1)[0] + 1]

    med_interval = np.median(np.diff(pulse_starts))
    stim_grps = np.split(pulse_starts, np.where(np.diff(pulse_starts) > med_interval*2)[0]+1)

    stim_summary = []
    for grp in stim_grps:
        start_time = grp[0]
        end_time = grp[-1]
        num_pulses = len(grp)
        duration = end_time - start_time
        if num_pulses > 1:
            mean_interval = np.mean(np.diff(grp))
        else:
            mean_interval = np.nan
        stim_summary.append({
            'stim_times': grp,
            'start_time': start_time,
            'end_time': end_time,
            'num_pulses': num_pulses,
            'duration': duration,
            'mean_interval': mean_interval
        })

    stim_df = pd.DataFrame(stim_summary)
    stim_df.index.name = 'stim_group'
    return stim_df



def summarize_antidromic(times, ap_path, lf_path):
    lfp_mean, csd_mean = evoked_lfp(lf_path, times)
    ap_env = evoked_ap(ap_path, times)
    return {'ap': ap_env, 'lfp': lfp_mean, 'csd': csd_mean}