# Utility functions for working with SpikeGLX files
# Author: Drew Headley
# Date: 2024-05-19

import os
import sys
import numpy as np
import re
sys.path.append('../code/')

def read_meta(meta_path):
    """
    Read the .meta file from a SpikeGLX recording
    
    Parameters
    ----------
    meta_path : str
        Path to the .meta file
        
    Returns
    -------
    meta : dict
        Dictionary containing the metadata
    """

    meta = {}
    with open(meta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            key, value = line.split('=')

            if not key.startswith('imDat'): # leave version numbers as strings
                # Format the values
                if re.match(r'^\d+$', value): # format an integer
                    value = int(value)
                elif re.match(r'^\d+\.\d+$', value): # format a decimal number
                    value = float(value)
                elif re.match(r'^-?\d+(?:,-?\d+)*$', value): # format a list of integers
                    value = [int(v) for v in value.split(',')]
                elif value.lower() == 'true': # format a boolean
                    value = True
                elif value.lower() == 'false': # format a boolean
                    value = False

            meta[key] = value
    
    return meta



def get_meta_path(bin_path):
    """
    Get the path to the .meta file from a SpikeGLX binary file.
    Tests if the .meta file exists in the same directory as the binary file.
    
    Parameters
    ----------
    bin_path : str
        Path to the binary file
        
    Returns
    -------
    meta_path : str
        Path to the .meta file
    """

    # check if bin path is a .bin file
    if not bin_path.endswith('.bin'):
        raise ValueError('Binary file must have .bin extension')

    # remove the .bin extension, add .meta extension
    meta_path = bin_path[:-4] + '.meta'

    # check if meta file exists
    if not os.path.exists(meta_path):
        raise FileNotFoundError('Meta file {} not found'.format(meta_path))
    
    return meta_path