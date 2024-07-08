# Functions for curve fitting
# Author: Drew Headley
# Date: 2024-06-13

from scipy.optimize import curve_fit
import numpy as np

# half-gaussian fit

def half_gauss(x, amp, sd, offset):
    """
    Curve for a positive sided gaussian curve.
    
    Parameters
    ----------
    x : array-like
        The x values to evaludate the curve at.
    amp : float
        The amplitude of the curve.
    sd : float
        The standard deviation of the curve.
    offset : float
        The offset of the curve (mean).
        
    Returns
    -------
    y : array-like
        The y value of the curve at the given x values.
    """

    y = amp*np.exp(-((x-offset)**2)/(2*sd**2))*(x>offset)
    return y

def fit_half_gauss(x, xp, yp, **kwargs):
    """
    Fit a half-gaussian curve to the data.

    Parameters
    ----------
    x : array-like
        The x values to evaludate the curve at.
    xp : array-like
        The x values of the data to fit.
    yp : array-like
        The y values of the data to fit.
    kwargs : dict
        Additional arguments to pass to curve_fit.

    Returns
    -------
    fit : array-like
        The y values of the curve at the given x values.
    popt : array-like
        The optimized parameters of the curve.
    """

    popt, _ = curve_fit(half_gauss, xp, yp, **kwargs)
    fit = half_gauss(x, *popt)
    return fit, popt