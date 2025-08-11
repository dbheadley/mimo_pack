# Functions for cross-correlation analyses
# Author: Drew Headley
# Date: 2025-07-11

import numpy as np
import pynapple as nap

def cc_to_prob(cc, counts=None, norm=True):
    """
    Convert cross-correlogram to probability conditioned on the reference spike.

    Parameters
    ----------
    cc : pandas.DataFrame
        Cross-correlogram DataFrame returned by pynapple.compute_correlogram.
    counts : pandas.DataFrame, optional
        Spike counts, indexed by spike group IDs. Only used if norm is True.
    norm : bool, optional
        Indicates whether the computed cross-correlogram was normalized by the
        target unit's firing rate. By default, Pynapple normalizes the cross-correlogram.

    Returns
    -------
    pandas.DataFrame
        Probability conditioned on the reference spike.
    """

    cc_prob = cc.copy()
    
    bin_width = np.median(np.diff(cc.index))
    if norm:
        if counts is None:
            raise ValueError("counts must be provided if norm is True.")

        for i in range(len(cc.columns)):
            target_id = cc.columns[i][1]
            cc_prob.iloc[:, i] = cc.iloc[:, i] * counts[target_id].values * bin_width
    else:
        for i in range(len(cc.columns)):
            cc_prob.iloc[:, i] = cc.iloc[:, i] * bin_width

    return cc_prob

def cc_correct_poly(cc, exclude_lag=0.005, order=12):
    """
    Corrects the cross-correlogram by fitting a polynomial to the baseline and subtracting it.

    Parameters
    ----------
    cc : nap.Ts
        Cross-correlogram to be corrected.
    exclude_lag : float, optional
        Time lag to exclude from the polynomial fit.
    order : int, optional
        Order of the polynomial to fit.

    Returns
    -------
    cc_corr : nap.Ts
        Corrected cross-correlogram.
    cc_baseline : nap.Ts
        Baseline polynomial fit.
    """
    lags = cc.index.values
    w = (np.abs(lags) > exclude_lag).astype(float)
    cent_lags = np.abs(lags) <= 0.0006
    w[cent_lags] = (w.size-np.sum(w))/(np.sum(cent_lags))

    cc_corr = cc.copy()
    cc_baseline = cc_corr.copy()
    for i in range(len(cc_corr.columns)):        
        # Fit polynomial to the baseline
        p = np.polyfit(lags, cc_corr.iloc[:, i], order, w=w)
        baseline = np.polyval(p, lags)
        cc_corr.iloc[:, i] -= baseline
        cc_baseline.iloc[:, i] = baseline
    
    return cc_corr, cc_baseline

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from mimo_pack.math.curvefitting import fit_sine_freq

    def simulate_and_test(mod_freqs, trans_probs, max_time=1000, dt=0.00003, f_rate=10):
        results_attenuation = []
        results_peak_prob = []
        results_actual_prob = []

        for mod_freq in mod_freqs:
            for trans_prob in trans_probs:
                print(f"Simulating for modulation frequency {mod_freq} Hz and transmission probability {trans_prob}")
                # Generate spike trains
                t = np.arange(0, max_time, dt)
                rate1 = f_rate * dt * np.ones_like(t)
                spks1 = t[np.where(np.random.poisson(rate1) > 0)[0]]

                rate2 = f_rate * (1 + np.sin(2 * np.pi * mod_freq * t)) * dt
                spks2 = t[np.where(np.random.poisson(rate2) > 0)[0]]

                spks1_sub = np.random.choice(spks1, size=int(len(spks1) * trans_prob), replace=False)
                spks3 = np.sort(np.concatenate((spks2, spks1_sub + 0.002)))

                spks1_pos = spks1[np.where(np.mod(spks1, 1/mod_freq) < 1/(2*mod_freq))[0]]
                spks1_neg = spks1[np.where(np.mod(spks1, 1/mod_freq) >= 1/(2*mod_freq))[0]]

                # Convert to nap.Ts objects
                spks1_ts = nap.Ts(t=spks1, time_units='s')
                spks2_ts = nap.Ts(t=spks2, time_units='s')
                spks3_ts = nap.Ts(t=spks3, time_units='s')
                spks1_pos_ts = nap.Ts(t=spks1_pos, time_units='s')
                spks1_neg_ts = nap.Ts(t=spks1_neg, time_units='s')

                spks_grp = nap.TsGroup({0: spks1_ts, 1: spks1_pos_ts, 2: spks1_neg_ts, 3: spks3_ts})

                # Compute cross-correlogram
                cc = nap.compute_crosscorrelogram(spks_grp, binsize=0.0005, windowsize=0.02, norm=False)

                # Convert to probability and correct
                cc_prob = cc_to_prob(cc, norm=False)
                cc_corr, _ = cc_correct_poly(cc_prob, order=12)

                # Power at modulation frequency before and after correction
                lags = cc.index.values
                exclude_lag = 0.005  # Exclude lags within this range for polynomial fitting
                mask = np.abs(lags) > exclude_lag
                sin_orig, p_orig = fit_sine_freq(lags, lags[mask], cc_prob[(2,3)].values[mask], mod_freq)
                sin_corr, p_corr = fit_sine_freq(lags, lags[mask], cc_corr[(2,3)].values[mask], mod_freq)
                attenuation = p_corr['amplitude'] / p_orig['amplitude']
                results_attenuation.append((mod_freq, trans_prob, attenuation))

                # Peak transmission probability in corrected cc
                peak_prob = np.max(cc_corr[(2,3)].values)
                results_peak_prob.append((mod_freq, trans_prob, peak_prob))
                results_actual_prob.append(trans_prob)

        # Plot results
        results_attenuation = np.array(results_attenuation)
        results_peak_prob = np.array(results_peak_prob)
        results_actual_prob = np.array(results_actual_prob)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Plot attenuation at modulation frequency
        for mod_freq in mod_freqs:
            mask = results_attenuation[:, 0] == mod_freq
            axs[0].plot(results_attenuation[mask, 1], results_attenuation[mask, 2], label=f"Mod Freq {mod_freq} Hz")
        axs[0].set_ylim(-0.1, 1.1)
        axs[0].set_xlabel("Transmission Probability")
        axs[0].set_ylabel("Attenuation at Modulation Frequency")
        axs[0].legend()
        axs[0].set_title("Attenuation at Modulation Frequency")

        # Plot peak transmission probability vs actual
        for mod_freq in mod_freqs:
            mask = results_peak_prob[:, 0] == mod_freq
            axs[1].plot(results_actual_prob[mask], results_peak_prob[mask, 2], label=f"Mod Freq {mod_freq} Hz")
        axs[1].plot([-1, 1], [-1, 1], 'k--')  # y=x line for reference
        axs[1].set_xlim(-0.01, 0.11)
        axs[1].set_ylim(-0.01, 0.11)
        axs[1].set_xlabel("Actual Transmission Probability")
        axs[1].set_ylabel("Peak Transmission Probability (Corrected CC)")
        axs[1].legend()
        axs[1].set_title("Peak Transmission Probability vs Actual")

        plt.tight_layout()
        plt.show()

    mod_freqs = [10, 16, 20, 50, 60, 80, 100]
    trans_probs = np.linspace(0, 0.1, 6)
    simulate_and_test(mod_freqs, trans_probs)