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


def piecewise_table_monotonic(vals, thresh=None):
    """
    Reduce a list of monotonically increasing values to a table of key 
    points that define a piecewise function. The rate of increase is assumed 
    to be the median difference between adjacent values. Error in the values that 
    would be returned by a piecewise linear interpolation function is kept 
    below an error threshold.
    
    Parameters
    ----------
    vals : N-length numpy array
        The values to reduce. 
        
    Optional
    --------
    threshold : float
        Maximum allowable error in the piecewise interpolation. Default
        is half the median difference between adjacent values.
        
    Returns
    -------
    table : Kx2 numpy array
        The key points of the piecewise function. The first column is the index
        of the keypoint in vals, and the second column is the value in vals at 
        that index. K is the number of key points found. There is always at least
        two, for the start and end of vals.


    Example
    -------
    import matplotlib.pyplot as plt
    test = np.array([np.nan, 1,2,3,4,5,6,8,9,11,12,13,14,15,16,17,
                     np.nan,np.nan,19,20])
    table = piecewise_table_monotonic(test)
    plt.plot(test)
    plt.plot(table[:,0], table[:,1], 'ro')
    """

    vals_num = len(vals)
    table = np.array([[0, vals[0]]])
    slope = np.nanmedian(np.diff(vals))
    if thresh is None:
        thresh = slope/2

    # Find the starts and ends of NaN values so that all of them can be included in the table
    nan_edges = np.argwhere(np.abs(np.diff(np.isnan(vals)))==1)
    if len(nan_edges) > 0:
        nan_edges = nan_edges.flatten()[np.newaxis,:]+1
        table = np.vstack((table, np.vstack((nan_edges, vals[nan_edges])).T, 
                           np.vstack((nan_edges-1, vals[nan_edges-1])).T))
                
    ind = 0
    while ind < vals_num-1:
        if np.isnan(vals[ind]):
            next_ind = np.argwhere(~np.isnan(vals[ind:]))
        else:
            pred_vals = vals[ind] + slope*np.arange(0, ((vals_num-1)-ind))
            cum_err = np.abs(np.nancumsum(vals[ind:-1]-pred_vals))
            next_ind = np.argwhere(cum_err>thresh)

        if len(next_ind) == 0:
            break
        else:
            next_ind = next_ind[0][0]

        if (ind>0):
            table = np.vstack((table, [ind+next_ind-1, vals[ind+next_ind-1]]))
        table = np.vstack((table, [ind+next_ind, vals[ind+next_ind]]))
        ind += next_ind
    
    table = np.vstack((table, [vals_num-1, vals[-1]]))
    uniq_inds = np.unique(table[:,0], return_index=True)[1]
    table = table[uniq_inds]
    return table


def exp_knee(x, offset, knee, exp):
    """
    Exponential knee function.
    Based on function from FOOOF package.
    
    Parameters
    ----------
    x : array-like
        The x values to evaluate the curve at.
    offset : float
        The offset of the curve.
    knee : float
        The knee point of the curve.
    exp : float
        The exponential factor of the curve.
        
    Returns
    -------
    y : array-like
        The y value of the curve at the given x values.
    """
    
    return offset - np.log10(knee + x**exp)

def fit_exp_knee(x, xp, yp, suppress_error=False, **kwargs):
    """
    Fit an exponential knee curve to the data.
    When working with spectral data, x values should
    be linearly spaced frequencies, and y values should
    be the log10 power spectral density values.

    Parameters
    ----------
    x : array-like
        The x values to evaluate the curve at.
    xp : array-like
        The x values of the data to fit.
    yp : array-like
        The y values of the data to fit.
    suppress_error : bool, optional
        If True, suppresses errors during fitting. 
        Default is False, which raises an error if fitting fails.
    kwargs : dict
        Additional arguments to pass to curve_fit.

    Returns
    -------
    fit : array-like
        The y values of the curve at the given x values.
    popt : array-like
        The optimized parameters of the curve.
    """

    # initial guess for the parameters
    if 'p0' not in kwargs:
        offset_guess = yp[np.where(xp == np.min(xp))[0][0]]
        knee_guess = 0.1
        exp_guess = 1.0
        kwargs['p0'] = [offset_guess, knee_guess, exp_guess]

    if suppress_error:
        try:
            popt, _ = curve_fit(exp_knee, xp, yp, **kwargs)
        except Exception as e:
            print(f"Error fitting exp_knee: {e}")
            popt = np.array([np.nan, np.nan, np.nan])
    else:
        popt, _ = curve_fit(exp_knee, xp, yp, **kwargs)

    fit = exp_knee(x, *popt)
    return fit, popt
