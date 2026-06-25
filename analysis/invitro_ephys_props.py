# Functions for measuring a neuron's 
# electrophysiological properties from in vitro recordings.
# Author: Drew Headley
# Date: 2025-07-12

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

AP_WIN_START = 0.0005  # seconds
AP_WIN_STOP = 0.003     # seconds
ASYMP_DUR = 0.1 # seconds, duration of the asymptotic phase at end of pulse
SAG_DUR = 0.1 # seconds, duration of the sag phase at start of pulse

def extract_iv_properties(iv_xr):
    """
    Extract electrophysiological properties from an IV protocol xarray DataArray.
    Returns a dictionary of properties similar to the provided MATLAB code.

    Parameters
    ----------
    iv_xr : xarray.DataArray
        The input data array containing voltage and current traces from an IV protocol,
        with required attributes such as 'sample_rate' and 'step_dur', and coordinate
        information for channels and time.
    Returns
    -------
    props : dict
        A dictionary containing extracted electrophysiological properties, including:
            - stim_start_s: float, stimulation start time (seconds)
            - stim_stop_s: float, stimulation stop time (seconds)
            - pulse_dur_s: float, pulse duration (seconds)
            - current_baselines_pA: array-like, baseline holding currents (pA)
            - current_steps_pA: array-like, injected current steps (pA)
            - rest_vm_mV: array-like, resting membrane potentials (mV)
            - rest_vm_mean_mV: float, mean resting membrane potential (mV)
            - rest_vm_sd_mV: float, standard deviation of resting membrane potential (mV)
            - ap_peak_times_s: array-like, action potential peak times (seconds)
            - ap_peak_amps_mV: array-like, action potential peak amplitudes (mV)
            - ap_waveforms_mV: array-like, action potential waveforms (mV)
            - ap_counts: array-like, number of action potentials per trial
            - ap_counts_reb: array-like, number of action potentials in rebound period
            - ap_threshold_pA: float, minimum current for action potential threshold (pA)
            - ap_threshold_mV: float, voltage at action potential threshold (mV)
            - ap_first_waveform_mV: array-like, waveform of the first action potential (mV)
            - ap_rates_hz: array-like, firing rates (Hz)
            - adaptation_ratio: float, adaptation ratio of firing rates
            - ap_halfwidth_s: float, half-width of the first action potential (seconds)
            - ap_peak_to_trough_s: float, peak-to-trough duration of the first action potential (seconds)
            - mem_v_mV: array-like, membrane voltage responses (mV)
            - mem_r_MOhm: float, membrane resistance (MΩ)
            - mem_t_s: float, membrane time constant (seconds)
            - sag_amp_mV: array-like, sag amplitude (mV)
            - sag_ratio: array-like, sag ratio
    Notes
    -----
    Requires helper functions:
        - _get_resting_potential
        - _detect_aps
        - _first_spike_properties
        - _rate_properties
        - _ap_shape_properties
        - _passive_properties
        - _sag_ratio
    The function is designed to mimic the output structure of similar MATLAB code for IV protocol analysis.
    """
    
    props = {}

    v_ch = int(np.where(iv_xr.ch_units.values == 'mV')[0][0])
    #i_ch = int(np.where(iv_xr.ch_units.values == 'pA')[0][0])
    v_traces = iv_xr.sel(channel=v_ch)
    #num_trials = v_traces.shape[1]

    stim_start = 0
    dt = 1.0 / float(iv_xr.attrs['sample_rate'])
    pulse_dur = float(iv_xr.attrs['step_dur'])
    stim_stop = stim_start + pulse_dur
    props['stim_start_s'] = stim_start
    props['stim_stop_s'] = stim_stop
    props['pulse_dur_s'] = pulse_dur

    v_pre = v_traces.sel(time=slice(None, 0)).to_numpy()
    v_dur = v_traces.sel(time=slice(0, pulse_dur)).to_numpy()
    v_reb = v_traces.sel(time=slice(pulse_dur+dt, pulse_dur+0.2)).to_numpy()

    baselines = iv_xr.hold_current.values
    pulse_vals = iv_xr.inj_current.values
    props['current_baselines_pA'] = baselines
    props['current_steps_pA'] = pulse_vals

    rest_vm_vals, rest_vm_mean, rest_vm_sd = _get_resting_potential(v_pre)
    props['rest_vm_mV'] = rest_vm_vals
    props['rest_vm_mean_mV'] = rest_vm_mean
    props['rest_vm_sd_mV'] = rest_vm_sd

    ap_peak_times, ap_peak_amps, ap_counts, ap_waveforms = _detect_aps(
        v_dur, dt
    )
    props['ap_peak_times_s'] = ap_peak_times
    props['ap_peak_amps_mV'] = ap_peak_amps
    props['ap_waveforms_mV'] = ap_waveforms
    props['ap_counts'] = ap_counts

    _, _, ap_counts, _ = _detect_aps(v_reb, dt)
    props['ap_counts_reb'] = ap_counts

    ap_min_i, ap_min_v, ap_first_wave = _first_spike_properties(
        ap_waveforms, pulse_vals
    )
    props['ap_threshold_pA'] = ap_min_i
    props['ap_threshold_mV'] = ap_min_v
    props['ap_first_waveform_mV'] = ap_first_wave

    ap_rates, adaptation_ratio = _rate_properties(ap_peak_times)
    props['ap_rates_hz'] = ap_rates
    props['adaptation_ratio'] = adaptation_ratio

    ap_halfwidth, ap_peak_to_trough = _ap_shape_properties(ap_first_wave)
    props['ap_halfwidth_s'] = ap_halfwidth
    props['ap_peak_to_trough_s'] = ap_peak_to_trough

    mem_v, mem_r, mem_t = _passive_properties(v_dur, pulse_vals, dt)
    props['mem_v_mV'] = mem_v
    props['mem_r_MOhm'] = mem_r * 1e-6 # convert to MOhms
    props['mem_t_s'] = mem_t

    sag_amp, sag_ratio = _sag_ratio(v_dur, pulse_vals, dt)
    props['sag_amp_mV'] = sag_amp
    props['sag_ratio'] = sag_ratio

    return props

def _get_resting_potential(v_traces):
    """
    Calculate the resting membrane potential statistics from voltage traces.

    Parameters
    ----------
    v_traces : np.ndarray
        Array of voltage traces, where each row represents a trace.

    Returns
    -------
    rest_vm_vals : np.ndarray
        Median voltage values across traces for each time point.
    rest_vm_mean : float
        Mean of the median resting membrane potentials.
    rest_vm_sd : float
        Standard deviation of the median resting membrane potentials (sample SD, ddof=1).
    """
    rest_vm_vals = np.median(v_traces, axis=0)
    rest_vm_mean = np.mean(rest_vm_vals)
    rest_vm_sd = np.std(rest_vm_vals, ddof=1)
    return rest_vm_vals, rest_vm_mean, rest_vm_sd

def _detect_aps(v_traces, dt):
    """
    Detects action potentials (APs) in voltage traces and extracts their properties.
    Parameters
    ----------
    v_traces : np.ndarray
        2D array of voltage traces with shape (num_timepoints, num_trials).
    dt : float
        Time step between samples in seconds.
    Returns
    -------
    ap_peak_times : list of np.ndarray
        List containing arrays of AP peak times (in seconds) for each trial.
    ap_peak_amps : list of np.ndarray
        List containing arrays of AP peak amplitudes for each trial.
    ap_counts : list of int
        List containing the number of APs detected in each trial.
    ap_waveforms : list of np.ndarray
        List containing arrays of AP waveforms for each trial, 
        each array has shape (window_length, num_aps).
    """
    ap_peak_times = []
    ap_peak_amps = []
    ap_waveforms = []
    ap_counts = []
    num_trials = v_traces.shape[1]
    ap_win_start = int(AP_WIN_START / dt)
    ap_win_stop = int(AP_WIN_STOP / dt)
    ap_window = np.arange(-ap_win_start, ap_win_stop+1)
    for j in range(num_trials):
        curr_v = v_traces[:, j]
        peaks, _ = find_peaks(curr_v, height=0, distance=int(0.001/dt))
        curr_v = np.pad(curr_v, (ap_win_start, ap_win_stop), 
                        mode='constant', constant_values=np.nan)
        ap_peak_times.append(peaks * dt)
        ap_peak_amps.append(curr_v[peaks+ap_win_start])
        ap_counts.append(len(peaks))

        wf = []
        for ind in peaks:
            inds = ap_window + ind + ap_win_start
            wf.append(curr_v[inds])
        ap_waveforms.append(np.array(wf).T)

    return ap_peak_times, ap_peak_amps, np.array(ap_counts), ap_waveforms


def _first_spike_properties(ap_waveforms, pulse_vals):
    """
    Extracts properties of the first detected action potential (spike) from a set of waveforms.

    Parameters
    ----------
    ap_waveforms : list or array-like
        A list or array of action potential waveforms for each trial. Each element should be a 2D array
        where rows correspond to waveform samples and columns to spikes.
    pulse_vals : list or array-like
        A list or array of pulse values corresponding to each trial.

    Returns
    -------
    min_i : float or np.nan
        The pulse value for the trial containing the first spike, or np.nan if no spikes are found.
    min_v : float or np.nan
        The initial voltage value of the first spike waveform, or np.nan if no spikes are found.
    ap_first_wave : array-like or np.nan
        The waveform of the first detected spike, or np.nan if no spikes are found.
    """

    first_spike_trial = [i for i, ap in enumerate(ap_waveforms) if ap.size > 0][0]
    if first_spike_trial is None:
        return np.nan, np.nan, np.nan
    min_i = pulse_vals[first_spike_trial]
    ap_first_wave = ap_waveforms[first_spike_trial][:,0]
    min_v = ap_first_wave[0]
    return min_i, min_v, ap_first_wave

def _rate_properties(ap_peak_times):
    """
    Calculate firing rate properties from action potential peak times.
    Parameters
    ----------
    ap_peak_times : list of array-like
        A list where each element contains the times (in seconds) of action potential
        for a trial.
    Returns
    -------
    ap_rates : list of float
        List of mean instantaneous firing rates (Hz) for each trial. If a trial has fewer 
        than 2 action potentials, returns np.nan for that trial.
    adaptation_ratio : float
        The adaptation ratio for the first trial with at least 4 action potentials, calculated 
        as the ratio of the first to last instantaneous firing rate.
        Returns np.nan if no such trial exists.
    """
    ap_rates = []

    # Calculate instantaneous firing rate
    for i, times in enumerate(ap_peak_times):
        if len(times) < 2:
            ap_rates.append(np.nan)
            continue
        
        ap_inst_rate = 1/np.diff(times)
        ap_rates.append(np.mean(ap_inst_rate))

    # Calculate adaptation ratio
    try:
        adapt_trial = [i for i, r in enumerate(ap_peak_times) if len(r) >= 4][0]
        ap_inst_rate = 1/np.diff(ap_peak_times[adapt_trial])
        adaptation_ratio = ap_inst_rate[0] / ap_inst_rate[-1]
    except IndexError:
        adaptation_ratio = np.nan
    
    return ap_rates, adaptation_ratio

def _ap_shape_properties(ap_wave):
    ap_halfwidth = np.nan
    ap_peak_to_trough = np.nan
    if not np.any(np.isnan(ap_wave)):
        t_wave = np.linspace(0.0005, 0.003, ap_wave.size)
        interp_t = np.arange(0.0005, 0.003, 0.00001)
        interp_func = interp1d(t_wave, ap_wave, kind='cubic', fill_value='extrapolate')
        up_wave = interp_func(interp_t)
        v_half = (np.max(up_wave) - up_wave[0]) / 2 + up_wave[0]
        ap_halfwidth = np.sum(up_wave > v_half) * 0.00001
        peak_ind = np.argmax(up_wave)
        ap_peak_to_trough = np.argmin(up_wave[peak_ind:]) * 0.00001
        
    return ap_halfwidth, ap_peak_to_trough

def _passive_properties(v_traces, pulse_vals, dt):
    """
    Calculates passive electrophysiological properties from voltage traces and current pulses.
    Parameters
    ----------
    v_traces : np.ndarray
        2D array of voltage traces (time x pulses).
    pulse_vals : np.ndarray
        1D array of current pulse amplitudes corresponding to each pulse.
    dt : float
        Time step between samples in seconds.
    Returns
    -------
    mem_v : np.ndarray
        Median steady-state membrane voltage for each pulse.
    mem_r : float or np.nan
        Estimated membrane resistance in Ohms, calculated from negative current injections.
    mem_t : float or np.nan
        Estimated membrane time constant (tau) in seconds, calculated from exponential fit to voltage response.
    """
    asymp_ind = int(ASYMP_DUR / dt)
    mem_r = np.nan
    mem_t = np.nan
    mem_v = np.median(v_traces[-asymp_ind:, :], axis=0)
    sel_neg_pulse = np.where(pulse_vals < 0)[0][-2]
    
    # Calculate membrane resistance as the slope of the voltage change
    # over the current change during negative current injections
    if sel_neg_pulse > 1:
        dv = np.diff(mem_v[(sel_neg_pulse-2):(sel_neg_pulse+2)]) * 0.001  # V
        di = np.diff(pulse_vals[(sel_neg_pulse-2):(sel_neg_pulse+2)]) * 1e-12     # A
        mem_r = np.mean(dv / di) # Ohms

    # Calculate membrane capacitance as the time constant of the voltage change
    # fit an exponential decay to the second to last negative pulse, capacitance is C = tau / R
    if sel_neg_pulse > 1 and mem_r > 0:
        v_fit = v_traces[:, sel_neg_pulse]
        t_fit = np.arange(0, v_fit.size * dt, dt)

        # tau is the time constant, a is the amplitude, b is the offset
        fit_func = lambda t, tau, a, b: (a*(1-np.exp(-t / tau)))+b
        p0 = [0.01, # 10 ms time constant
              pulse_vals[sel_neg_pulse]*mem_r*1e-9, # expected amplitude in mV
              v_fit[0]] # initial voltage
        bounds = ([0, -np.inf, -200], [1, 0, 0])
        popt, _ = curve_fit(fit_func, t_fit, v_fit, p0=p0, bounds=bounds)
        tau = popt[0]
        mem_t = tau  # Time constant in seconds

    return mem_v, mem_r, mem_t

def _sag_ratio(v_traces, pulse_vals, dt):
    """
    Calculates the sag amplitude and sag ratio from voltage traces in response 
    to hyperpolarizing current pulses.
    
    Parameters
    ----------
    v_traces : np.ndarray
        2D array of voltage traces (time x trials).
    pulse_vals : np.ndarray
        1D array of pulse amplitudes for each trial.
    dt : float
        Time step (sampling interval) in milliseconds.
    
    Returns
    -------
    sag_amp : float
        The sag amplitude, defined as the difference between the minimum voltage during 
        the sag and the asymptotic voltage.
    sag_ratio : float
        The sag ratio, defined as the ratio of the asymptotic voltage to the minimum 
        sag voltage.
    """
    asymp_ind = int(ASYMP_DUR / dt)
    sag_ind = int(SAG_DUR / dt)
    neg_trial = np.argmin(pulse_vals)

    v_resp = v_traces[:, neg_trial] - v_traces[0, neg_trial]  # subtract resting potential
    v_sag = np.min(v_resp[0:sag_ind])
    v_asymp = np.median(v_resp[-asymp_ind:])
    sag_amp = v_sag - v_asymp
    sag_ratio = v_asymp / v_sag
    return sag_amp, sag_ratio