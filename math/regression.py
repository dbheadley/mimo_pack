# Functions for linear regression
# Author: Drew Headley
# Date: 2025-07-22

import numpy as np
from tqdm import tqdm

def rolling_linear_regression(y : np.ndarray, x : np.ndarray, window_size : int, 
                              verbose=False) -> tuple:
    """
    Performs rolling linear regression of y onto x over a specified window size.

    This function calculates the linear regression (y = mx + c) for rolling
    windows of the input data. It returns the fitted values, residuals, and
    the regression coefficients (offset and slope) for each window.

    Parameters
    ----------
    y : np.ndarray
        A 1D numpy array of the dependent variable.
    x : np.ndarray
        A 1D numpy array of the independent variable. Must be the
        same size as y.
    window_size : int
        The number of data points to include in each regression window.

    Returns
    -------
    fitted_values : np.ndarray
        A 1D array of the same size as the inputs.
        It contains the value of the regression line at each point x,
        calculated from the model for the window ending at that point.
        The first (window_size - 1) values are NaN.
    residuals : np.ndarray
        A 1D array of the same size as the inputs,
        representing the difference (y - fitted_values). The first
        (window_size - 1) values are NaN.
    coefficients : np.ndarray
        A 2D array of shape (n, 2) where n is the
        size of the input arrays. The first column is the regression
        offset (intercept) and the second column is the slope. The first
        (window_size - 1) rows are NaN.
    """

    # --- Input Validation ---
    if not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray):
        raise TypeError("Inputs 'y' and 'x' must be numpy arrays.")
    if y.ndim != 1 or x.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional.")
    if y.shape != x.shape:
        raise ValueError("Input arrays 'y' and 'x' must have the same shape.")
    if not isinstance(window_size, int) or window_size <= 1:
        raise ValueError("'window_size' must be an integer greater than 1.")
    if window_size > len(y):
        raise ValueError("'window_size' cannot be larger than the data length.")

    n = len(y)

    # --- Pre-allocate Output Arrays with NaNs ---
    # This is more efficient than appending to lists
    fitted_values = np.full(n, np.nan)
    residuals = np.full(n, np.nan)
    coefficients = np.full((n, 2), np.nan) # Columns for [offset, slope]

    # --- Perform Rolling Regression ---
    # Start the loop from the first point where a full window is available
    if verbose:
        iter_obj = tqdm(range(window_size - 1, n), desc="Rolling Regression")
    else:
        iter_obj = range(window_size - 1, n)

    for i in iter_obj:
        # Define the current window for both x and y
        start_index = i - window_size + 1
        end_index = i + 1
        y_window = y[start_index:end_index]
        x_window = x[start_index:end_index]

        # Prepare the design matrix A for lstsq.
        # It needs to be a column of x values and a column of ones for the intercept.
        A = np.vstack([x_window, np.ones(window_size)]).T

        try:
            # Use numpy's least squares function to find the slope and offset
            # result[0] contains the coefficients [slope, offset]
            slope, offset = np.linalg.lstsq(A, y_window, rcond=None)[0]

            # Store the calculated coefficients
            # Note: we store as [offset, slope] as per the docstring
            coefficients[i] = [offset, slope]

            # Calculate the fitted value for the current point 'i'
            # using the model derived from the window ending at 'i'
            fitted_values[i] = slope * x[i] + offset

            # Calculate the residual for the current point 'i'
            residuals[i] = y[i] - fitted_values[i]

        except np.linalg.LinAlgError:
            # This can happen if the matrix A is singular, though unlikely with this setup.
            # We'll just let the NaN values remain for this window.
            print(f"Warning: Could not compute regression for window ending at index {i}.")
            continue

    return fitted_values, residuals, coefficients