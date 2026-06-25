# Short generic utility functions for working with numeric arrays
# Author: Drew Headley
# Date: 2024-06-13

import numpy as np

# Split an array into multiple sub-arrays that are copies of the original.
def split_copy(array, *args, **kwargs):
    """
    Split an array into multiple sub-arrays that are copies of the original.
    The arguments are the same as numpy.split, but the resulting arrays are copies
    of the original, not views. This is useful when you need to modify the sub-arrays
    without affecting the original array. The original array is not modified.
    
    Parameters
    ----------
    array : np.ndarray
        The array to split.
    *args : int
        The arguments to pass to np.split.
    **kwargs : dict
        The keyword arguments to pass to np.split.
        
    Returns
    -------
    list
        A list of the sub-arrays, each of which is a copy of the original.
    """
    
    return [np.copy(x) for x in np.split(array, *args, **kwargs)]