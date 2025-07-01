# Unit preprocessing related functions
# Author: Drew Headley
# Created: 2024-06-11

import os
import numpy as np
import dclut as dcl
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ..fileio.spikeglx import read_meta, get_meta_path
from ..plot.map import amp_map, wave_map
from ..plot.histogram import stairs_fl

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

def unit_occupancy(spk_t, start, end):
    """
    Calculate the occupancy of a unit in a given time period.

    Parameters
    ----------
    spk_t : np.ndarray
        Spike times in seconds.
    start : numeric
        Start time of the period in seconds.
    end : numeric
        End time of the period in seconds.

    Returns
    -------
    occupancy : float
        Occupancy of the unit in the period as a fraction of the total time.
    """
    
    total_dur = end - start # total duration of the period
    spk_dur = spk_t.max() - spk_t.min() # total duration of spiking in the period
    occupancy = spk_dur / total_dur # fraction of time the unit is occupied by spikes
    
    return float(occupancy)

def unit_refractory_violations(spk_t, ref_period=0.002, start=None, end=None):
    """
    Calculates the ratio of observed to predicted refractory period violations.
    Predicted violations are based on the mean firing rate. Can serve as a measure
    of the rate of false positive spikes.

    Parameters
    ----------
    spk_t : array
        The spike times, sorted in ascending order.
    ref_period : float, optional
        The refractory period in seconds. Default is 0.002 seconds (2 ms).
    start : float, optional
        Start time of the period in seconds. If None, uses the first spike time.
    end : float, optional
        End time of the period in seconds. If None, uses the last spike time.

    Returns
    -------
    r_fp : float
        The ratio of observed to predicted refractory period violations
    """
    
    if start is None:
        start = spk_t[0]
    if end is None:
        end_t = spk_t[-1]

    # ensure spike times are within the specified period
    spk_t = spk_t[(spk_t >= start) & (spk_t <= end)]
    if spk_t.size == 0:
        return np.nan

    num_spks = spk_t.size
    dur = end-start # total duration of spiking
    viol_count = np.sum(np.diff(spk_t)<=ref_period) # number of refractory period violations
    refract_time = 2*ref_period*num_spks # total potential time for refractory period violations
    spk_rate = num_spks/dur # mean firing rate, irrespective of refractory period
    viol_rate = viol_count/refract_time # firing rate just during the refractory period
    r_fp = viol_rate/spk_rate # ratio of observed to predicted violations
    return float(r_fp)


# based the algorithm from the Allen Institute spike quality metrics code
# which is based on a measure proposed in Hill et al. 2011 J Neurosci 31: 8699-8705
# I have modified it to account for the lost spikes when calculating the 
# total spike count
def amp_cutoff(spk_a):
    '''
    Calculates the amplitude cutoff for a given unit

    Parameters
    ----------
    spk_a : array
        The spike amplitudes for a unit

    Returns
    -------
    miss_prob : float
        The probability that a spike will be missed due to amplitude cutoff
    '''
    dist, bins = np.histogram(spk_a, bins=50)
    dist = gaussian_filter1d(dist, 3) # smooth the distribution
    peak_idx = np.argmax(dist) # find the peak

    # find the first point in the distribution above the peak that
    # falls below the probability density for the lowest amplitude.
    # If the true distribution of spike amplitudes is symmetric, then
    # then the area under from this point to the maximum amplitude should
    # be equal to the 
    g = np.argmin(np.abs(dist[peak_idx:]-dist[0]))+peak_idx

    # calculate the area under the curve from the end of the distribution
    miss_count = np.sum(dist[g:]) # calculate area under the curve from the end of the distribution

    total_count = miss_count+spk_a.size # to estimate total count, add the missed to the observed
    miss_prob = miss_count/total_count # get proportion of total spikes missed

    # have max_prob cutoff at 0.5
    miss_prob = np.min([miss_prob, 0.5])

    return float(miss_prob)

def waveform_spread(peak_map, x_pos, y_pos):
    """
    Calculates the spread of the waveform across channels
    by measuring the radius of the halfwidth.

    Parameters
    ----------
    peak_map : np.ndarray
        The peak values at each of the channel locations for the waveform.
    x_pos : np.ndarray
        The x coordinates of the channels.
    y_pos : np.ndarray
        The y coordinates of the channels.
    Returns
    -------
    spread : float
        The spread of the waveform across channels in microns.
    """

    peak_map_abs = np.abs(peak_map)  # get absolute values of peak map

    # get the peak channel
    peak_chan = np.argmax(peak_map_abs)
    peak_x = x_pos[peak_chan]
    peak_y = y_pos[peak_chan]

    # get the furthest channel whose peak value is above half the maximum
    half_max = np.max(peak_map_abs)/2
    above_half = np.where(peak_map_abs > half_max)[0]
    if above_half.size == 0:
        return 0.0  # no channels above half max, spread is zero
    
    chan_dists = np.sqrt((x_pos[above_half] - peak_x)**2 + 
                         (y_pos[above_half] - peak_y)**2)
    spread = np.max(chan_dists)  # maximum distance from peak channel to any channel above half max
    return float(spread)

def waveform_snr(waveforms, mode='resid', mean_waveform=None):
    """
    Calculate th signal-to-noise ratio (SNR) of a spike waveform.
    
    Parameters
    ----------
    waveforms : np.ndarray, (n_samples, n_channels, n_spikes)
        The spike waveforms.
    mode : str, optional
        The type of SNR to calculate. Options:
        'resid' - Residual SNR, ratio of the standard deviation of the 
        mean waveform to the standard deviation of the residuals. The 
        channel with the largest waveform is used. Inspired by 
        Joshua et al. 2007. Default is 'resid'.
    mean_waveform : np.ndarray, optional
        The mean waveform to use for calculating the residuals. If None, 
        it will be calculated from the waveforms. Default is None.

    Returns
    -------
    snr : float
        The signal-to-noise ratio of the spike waveform.
    """
    
    if waveforms.ndim != 3:
        raise ValueError('Waveforms must be a 3D array with shape ' +
                         '(n_samples, n_channels, n_spikes)')
    if mode == 'resid':
        # calculate the residuals of the waveforms
        if mean_waveform is None:
            mean_waveform = np.mean(waveforms, axis=2)

        # get channel with the largest waveform
        peak_chan = np.argmax(np.max(np.abs(mean_waveform), axis=0))
        peak_waveform = mean_waveform[:, peak_chan]

        # calculate the residuals
        residuals = waveforms[:, peak_chan, :] - peak_waveform[:, np.newaxis]

        # calculate the SNR
        mean_std = np.std(peak_waveform)
        resid_std = np.std(residuals)
        snr = mean_std / resid_std

    else:
        raise ValueError("Unsupported SNR type: {}".format(mode))
    
    return float(snr)

def pynapple_spikes_qc(spks_pyn, dcl_file, wave_num=1000, ref_period=0.002, snr_mode='resid',
                       wave_period=[-1.0, 2.0], report_path=None, verbose=False):
    """
    Perform quality control on spikes in a pynapple TsGroup object with
    a 'waveform' metadata field. Spikes must be from a single probe.
    
    Parameters
    ----------
    spks_pyn : pynapple.TsGroup
        The spike times and waveforms in a TsGroup object.
    dcl_file : str
        Full file path to a dclut json file with a 'time', 'ch_x', and 'ch_y' scale. 
        The dclut file will also be used to obtain spike waveforms and the geometry
        of the probe recording sites (based on 'ch_x' and 'ch_y').
    wave_num : int, optional
        The maximum number of spikes to sample for quality control. Default is 1000.
        If the number of spikes is less than this, all spikes will be used.
    ref_period : float, optional
        The refractory period in seconds. Default is 0.002 seconds (2 ms).
    snr_mode : str, optional
        The type of SNR to calculate. Default is 'resid'.
    wave_period : list, optional
        The time period in millisecond to use for extracting spike waveforms.
        Default is [-1.0, 2.0], which corresponds to 1 ms before and 2 ms after 
        the spike peak.
    report_path : str, optional
        Path to directory where to save a report of the quality control results.
        If None, no report will be saved. Default is None.
    verbose : bool, optional
        Whether to print progress messages. Default is False.

    Returns
    -------
    spks_pyn : pynapple.TsGroup
        The input TsGroup object with quality control metadata fields:
        Occupancy: float
            The occupancy of the unit in the period as a fraction of the total time.
        RefractoryViolations: float
            The ratio of observed to predicted refractory period violations.
        AmplitudeCutoff: float
            The probability that a spike will be missed due to amplitude cutoff.
        WaveformSpread: float
            The spread of the waveform across channels in electrode spatial units 
            (typically microns).
        WaveformSNR: float
            The signal-to-noise ratio of the spike waveform.
    """

    # check if the TsGroup has a 'waveform' metadata field
    if 'waveform' not in spks_pyn.metadata_columns:
        raise ValueError('TsGroup must have a "waveform" metadata field for quality control.')
    
    # load the dclut file
    if not os.path.isfile(dcl_file):
        raise FileNotFoundError('DCL file no found: {}'.format(dcl_file))
    
    # check if reports need to be saved
    if report_path is not None:
        # create the report path if it does not exist
        if not os.path.exists(report_path):
            os.makedirs(report_path)

    
    if verbose:
        print("Loading dclut file: {}".format(dcl_file))
    spks_dcl = dcl.dclut(dcl_file)
    times_sess = spks_dcl.scale_values(scale='time')
    samp_per_ms = np.ceil(((1/np.nanmedian(np.diff(times_sess))) / 1000)).astype(np.int64)
    wave_win = np.array([[wave_period[0]*samp_per_ms],
                         [wave_period[1]*samp_per_ms]]).astype(np.int64)
    
    x_pos = spks_dcl.scale_values(scale='ch_x')
    y_pos = spks_dcl.scale_values(scale='ch_y')

    if verbose:
        spks_iter = tqdm(spks_pyn, desc='Processing units')
    else:
        spks_iter = spks_pyn

    # recording edges to exclude when getting spike waveforms
    
    occupy = []
    ref_fp = []
    amp_cut = []
    wave_spread = []
    wave_snr = []

    # iterate over units in the TsGroup
    for unit_id in spks_iter:
        spks_dcl.reset()
        unit = spks_pyn[unit_id]
        times = unit.times()
        spike_num = times.size
        time_support = unit.time_support
        start_sess = time_support['start']
        end_sess = time_support['end']
        if 'class' in spks_pyn.metadata_columns:
            unit_class = spks_pyn.get_info('class')[unit_id]
        else:
            unit_class = 'Unknown'

        # get example spike waveforms
        if spike_num > wave_num:
            sample_times = np.sort(np.random.choice(times[(times>(start_sess-(wave_period[0]/1000))) &
                                                          (times<(end_sess-(wave_period[1]/1000)))], 
                                                          wave_num))
        else:
            sample_times = times[(times>(start_sess-(-wave_period[0]/1000))) &
                                 (times<(end_sess-(wave_period[1]/1000)))]
        # convert spike times to indices
        sample_inds = np.searchsorted(times_sess, sample_times).astype(np.int64)
        spks_dcl.intervals({'s0': (sample_inds+wave_win).T}, select_mode='split')
        
        try:
            waves = np.stack(spks_dcl.read(), axis=2)
        except Exception as e:
            import pdb; pdb.set_trace()

        # get mean waveform, subtract trend baseline from each spike
        # get map of the strength across channels, then identify 
        # the channel with the largest waveform
        waves = waves - np.linspace(waves[0,:,:], waves[-1,:,:], int(wave_win[1]-wave_win[0]))
        mean_wave = np.mean(waves, axis=2)
        amp_wave = np.linalg.norm(mean_wave, axis=0)
        peak_ind = np.argsort(amp_wave)[-1]  # index of the channel with the largest waveform
        peak_dists = (x_pos - x_pos[peak_ind])**2 + (y_pos - y_pos[peak_ind])**2
        near_inds = np.argsort(peak_dists)[:8]
        peak_wave = mean_wave[:, near_inds]

        # calculate occupancy
        occupy.append(unit_occupancy(times, start=start_sess, end=end_sess))

        # calculate refractory violations
        ref_fp.append(unit_refractory_violations(times, ref_period=ref_period, 
                                                 start=start_sess, end=end_sess))
        # waveform spread
        wave_spread.append(waveform_spread(amp_wave, x_pos, y_pos))

        # calculate amplitude cutoff
        # get scale factor required to fit normalized waveform to each sample waveform
        wave_norm = (peak_wave / (np.linalg.norm(peak_wave, axis=0)+1)).flatten() # +1 to avoid division by zero
        wave_2d = waves[:, near_inds, :].transpose((2,0,1))
        wave_amps = np.concatenate([np.linalg.lstsq(wave_norm.reshape(-1,1), 
                                                        wave.reshape(-1,1), rcond=-1)[0] 
                                        for wave in wave_2d])
        amp_cut.append(amp_cutoff(wave_amps))

        # calculate SNR
        wave_snr.append(waveform_snr(waves[:, near_inds, :], mode=snr_mode))

        if report_path is not None:
            fig = plt.figure(figsize=(8.5, 11))
            gs = GridSpec(6, 6, figure=fig)
            ax_tbl = fig.add_subplot(gs[0:3, 0:2])
            ax_waves = fig.add_subplot(gs[0:3, 2:4])
            ax_spread = fig.add_subplot(gs[0:3, 4:6])
            ax_firinghist = fig.add_subplot(gs[3, 0:6])
            ax_isi = fig.add_subplot(gs[4:6, 0:3])
            ax_amp = fig.add_subplot(gs[4:6, 3:6])

            # plot unit info table
            ax_tbl.axis('off')
            row_labels = ['Unit ID', 'Class', 'Occupancy', 'Refractory violations',
                          'Amplitude cutoff', 'Waveform spread', 'Waveform SNR']

            cell_text = [[f'{unit_id}'], [unit_class], [f'{occupy[-1]:.2f}'], 
                         [f'{ref_fp[-1]:.3f}'], [f'{amp_cut[-1]:.3f}'], 
                         [f'{wave_spread[-1]:.1f}'], [f'{wave_snr[-1]:.2f}']]
            ax_tbl.table(cellText=cell_text, rowLabels=row_labels, loc='center')

            # plot example waveforms
            wave_map(peak_wave, x_pos=x_pos[near_inds], y_pos=y_pos[near_inds],
                     ax=ax_waves, x_scale=0.3, y_scale=0.5)
            ax_waves.set_title('Example waveforms')

            # plot waveform spread
            amp_map(amp_wave, x_pos=x_pos, y_pos=y_pos, ax=ax_spread)
            ax_spread.set_title('Waveform spread')

            # plot firing rate histogram
            fr_hist, bins = np.histogram(times, bins=np.arange(start_sess, end_sess, 60))
            stairs_fl(fr_hist/60, bins, fill_color='black', edge_color='black', 
                      baseline=0, ax=ax_firinghist)
            ax_firinghist.set_title('Firing rate histogram')
            ax_firinghist.set_xlabel('Time (s)')
            ax_firinghist.set_ylabel('Firing rate (Hz)')
            ax_firinghist.set_xlim(start_sess, end_sess)

            # plot inter-spike interval histogram
            isi = np.diff(times)
            isi_hist, isi_bin = np.histogram(isi, bins=10**np.arange(-4, 1, 0.1),
                                             density=True)
            stairs_fl(isi_hist, isi_bin, fill_color='black', edge_color='black', 
                      baseline=0, ax=ax_isi)
            ax_isi.set_title('Inter-spike interval histogram')
            ax_isi.set_xlabel('ISI (s)')
            ax_isi.set_ylabel('Probability density')
            ax_isi.set_xscale('log')
            ax_isi.set_xlim(isi_bin[0], isi_bin[-1])

            # plot amplitude histogram
            amp_hist, amp_bins = np.histogram(wave_amps, bins=50, density=True)
            stairs_fl(amp_hist, amp_bins, fill_color='black', edge_color='black', 
                      baseline=0, ax=ax_amp)
            ax_amp.set_title('Amplitude histogram')
            ax_amp.set_xlabel('Amplitude (a.u.)')
            ax_amp.set_ylabel('Probability density')

            fig.tight_layout()

            # save the report
            report_file = os.path.join(report_path, f'unit_{unit_id}_qc_report.pdf')
            fig.savefig(report_file, bbox_inches='tight')
            plt.close(fig)

    # add quality control metadata to the TsGroup
    spks_pyn.set_info(Occupancy=occupy,
                      RefractoryViolations=ref_fp,
                      AmplitudeCutoff=amp_cut,
                      WaveformSpread=wave_spread,
                      WaveformSNR=wave_snr)

    return spks_pyn


    # wave_list = []
    # samp_num = 1000
    # ind_max = spks_dcl.dcl['file']['shape'][0]
    # wave_win = np.array([[-30], [60]])
    # x_pos = spks_dcl.scale_values(scale='ch_x')
    # y_pos = spks_dcl.scale_values(scale='ch_y')
    # for id in clu_iter:
    #     spks_dcl.reset()

    #     spk_inds = spk_inds_dict[id]
    #     spk_inds = spk_inds[spk_inds < (ind_max-60)]
    #     spk_inds = spk_inds[spk_inds > 30]
    #     num_spks = spk_inds.size

    #     if num_spks > samp_num:
    #         spk_inds = np.sort(np.random.choice(spk_inds, samp_num))

    #     # get spike waveforms
    #     spks_dcl.intervals({'s0': (spk_inds+wave_win).T}, select_mode='split')
    #     waves = np.stack(spks_dcl.read(), axis=2)

    #     # get mean spike waveform
    #     # subtract trend baseline from each spike
    #     waves = waves - np.linspace(waves[0,:,:], waves[-1,:,:], 90) 
    #     mean_wave = np.mean(waves, axis=2)

    #     # identify 8 channels near where the spike waveform is largest
    #     # get min or max (whicher is larger magnitude) for each channel
    #     peak_map = mean_wave[np.argmax(np.abs(mean_wave), axis=0), 
    #                             np.arange(mean_wave.shape[1])]
    #     wave_amp = np.linalg.norm(mean_wave, axis=0)
    #     peak_ind = np.argsort(wave_amp)[-1]
    #     peak_dists = (x_pos - x_pos[peak_ind])**2 + (y_pos - y_pos[peak_ind])**2
    #     near_inds = np.argsort(peak_dists)[:8]
        
    #     # sort near_inds by amplitude of the waveform
    #     near_inds = near_inds[np.argsort(wave_amp[near_inds])] 
    #     waveform = mean_wave[:, near_inds]

    #     # get scale factor required to fit normalized waveform to each sample waveform
    #     waveform_norm = (waveform / np.linalg.norm(waveform, axis=0)).flatten()
    #     waveform_2d = waves[:, near_inds, :].transpose((2,0,1))
    #     waveform_amps = np.concatenate([np.linalg.lstsq(waveform_norm.reshape(-1,1), 
    #                                                     wave.reshape(-1,1))[0] 
    #                                     for wave in waveform_2d])
        
    #     x_near = x_pos[near_inds]
    #     y_near = y_pos[near_inds]

    #     if waves.shape[2] > 100:
    #         wave_examples = waves[:, near_inds, :100]
    #     else:
    #         wave_examples = waves[:, near_inds, :]
        
    #     wave_list.append({'waveform': waveform, 'waveform_examples': wave_examples,
    #                         'inds': near_inds, 'x': x_near, 'y': y_near,'amps': waveform_amps,
    #                         'peak_map': peak_map, 'x_map': x_pos, 'y_map': y_pos})