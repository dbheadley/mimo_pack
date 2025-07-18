# Functions for cross-correlation analyses
# Author: Drew Headley
# Date: 2025-07-11

import numpy as np

def cc_ac_deconv(cc, ac1, ac2, sigma=0):
    """
    Correct for autocorrelation in cross-correlation

    Parameters
    ----------
    cc : array-like
        Cross-correlation values
    ac1 : array-like
        Autocorrelation of the first point process
    ac2 : array-like
        Autocorrelation of the second point process
    sigma : float, optional
        Standard deviation for Gaussian smoothing, by default 2

    Returns
    -------
    cc_corr : array-like
        Cross-correlation corrected for autocorrelation
    """

    # test whether ac1 and ac2 are the same length as cc
    if len(ac1) != len(cc) or len(ac2) != len(cc):
        raise ValueError("ac1 and ac2 must be the same length as cc")
    
    # ac1 = np.fft.ifftshift(ac1)
    # ac2 = np.fft.ifftshift(ac2)
    # cc = np.fft.ifftshift(cc)

    # convert to frequency domain
    ac1_fft = np.fft.rfft(ac1)
    ac2_fft = np.fft.rfft(ac2)
    cc_fft = np.fft.rfft(cc)

    # deconvolve cross-correlation by autocorrelations
    cc_fft /= np.sqrt(np.real(ac1_fft) * np.real(ac2_fft) + 1e-15)

    # apply Gaussian smoothing
    if sigma > 0:
        freqs = np.fft.fftfreq(len(cc))
        gauss_filter = np.exp(-(freqs**2)/(2 *(sigma/len(cc))**2))
        cc_fft *= gauss_filter

    
    cc_corr = np.fft.irfft(cc_fft)
    #cc_corr = np.fft.fftshift(cc_corr).real

    return cc_corr



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    sig1 = np.random.poisson(0.1, 10000)
    sig2 = np.concatenate([sig1[5:], np.zeros(5)]) + np.random.poisson(0.1, 10000)

    sig2_blanked = sig2 * np.tile(np.floor(np.arange(20)/10), 500)

    plt.figure()
    plt.plot(sig1[:100])
    plt.plot(sig2_blanked[:100])

    lags = np.arange(-20, 21)
    ac1 = np.zeros(lags.shape)
    ac2 = np.zeros(lags.shape)
    cc = np.zeros(lags.shape)

    for i, lag in enumerate(lags):
        if lag < 0:
            ac1[i] = np.dot(sig1[:lag], sig1[-lag:])
            ac2[i] = np.dot(sig2_blanked[:lag], sig2_blanked[-lag:])
            cc[i] = np.dot(sig1[:lag], sig2_blanked[-lag:])
        elif lag > 0:
            ac1[i] = np.dot(sig1[lag:], sig1[:-lag])
            ac2[i] = np.dot(sig2_blanked[lag:], sig2_blanked[:-lag])
            cc[i] = np.dot(sig1[lag:], sig2_blanked[:-lag])
        else:
            ac1[i] = np.dot(sig1, sig1)
            ac2[i] = np.dot(sig2_blanked, sig2_blanked)
            cc[i] = np.dot(sig1, sig2_blanked)

    cc_corr = cc_ac_deconv(cc, ac1, ac2)

    plt.figure(figsize=(12, 6))
    plt.subplot(4, 1, 1)
    plt.plot(lags, cc, label='Cross-correlation')
    plt.title('Cross-correlation')
    plt.subplot(4, 1, 2)
    plt.plot(lags, ac1, label='Autocorrelation 1')
    plt.title('Autocorrelation 1')
    plt.subplot(4, 1, 3)
    plt.plot(lags, ac2, label='Autocorrelation 2')
    plt.title('Autocorrelation 2')
    plt.subplot(4, 1, 4)
    plt.plot(lags, cc_corr, label='Corrected Cross-correlation')
    plt.title('Corrected Cross-correlation')
    plt.tight_layout()
    plt.show()

