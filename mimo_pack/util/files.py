# Helpful functions for working with files/directories
# Author: Drew Headley
# Date: 2024-07-15

import os
import re

def find_all_matching_files(directory, pattern):
    """
    Find all files in a directory and subdirectories that match a pattern.
    
    Parameters
    ----------
    directory : str
        The directory to search.
    pattern : str
        The pattern to match as a regular expression
        
    Returns
    -------
    matching_files : list
        A list of the file paths that match the pattern.
    """

    matching_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if re.search(pattern, file):
                matching_files.append(os.path.join(root, file))
    
    return matching_files