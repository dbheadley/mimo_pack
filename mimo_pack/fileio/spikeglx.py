# Utility functions for working with SpikeGLX files
# Author: Drew Headley
# Date: 2024-05-19

import os
import numpy as np
import pandas as pd
import re
from dclut import create_dclut

def read_meta(bin_path):
    """
    Read the .meta file associated with a SpikeGLX .bin file
    
    Parameters
    ----------
    meta_path : str
        Path to the .meta file
        
    Returns
    -------
    meta : dict
        Dictionary containing the metadata
    """

    check_bin_path(bin_path)

    meta_path = get_meta_path(bin_path)
    meta = {}
    with open(meta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue

            match = re.search(r"(.*?)=(.*)", line)   #line.split('=')
            key = match.group(1).strip()
            value = match.group(2).strip()

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

    # validate binary file path
    check_bin_path(bin_path)
    
    # remove the .bin extension, add .meta extension
    meta_path = bin_path[:-4] + '.meta'

    # check if meta file exists
    if not os.path.exists(meta_path):
        raise FileNotFoundError('Meta file {} not found'.format(meta_path))
    
    return meta_path

def get_bin_memmap(bin_path):
    """
    Get a memory map of the binary file.
    
    Parameters
    ----------
    bin_path : str
        Path to the binary file
    
    Returns
    -------
    bin_mm : np.memmap
        Memory map of the binary file
    """

    # validate binary file path
    check_bin_path(bin_path)
    
    bin_size = os.path.getsize(bin_path)
    meta = read_meta(bin_path)
    
    fs = meta['imSampRate']
    chan_num = meta['nSavedChans']
    samp_num = bin_size // (2 * chan_num)

    # open binary file, read only
    bin_mm = np.memmap(bin_path, dtype='int16', mode='r', shape=(samp_num, chan_num))

    return bin_mm


def check_bin_path(bin_path):
    """
    Check if the binary file exists and has a .bin extension.
    
    Parameters
    ----------
    bin_path : str
        Path to the binary file

    """

    # check if bin path is a .bin file
    if not bin_path.endswith('.bin'):
        raise ValueError('Binary file must have .bin extension')

    # check if the binary file exists
    if not os.path.exists(bin_path):
        raise FileNotFoundError('Binary file {} not found'.format(bin_path))
    

def get_geommap(bin_path):
    """
    Gets the geometry map of each channel on the probe from the .bin file

    Parameters
    ----------
    bin_path : str
        Path to the binary file

    Returns
    -------
    g_df : pd.DataFrame
        Geometry map of the recorded electrodes. Each row is a channel.
        The columns are: shank, x, y, used (u-flag)
    """
    

    meta = read_meta(bin_path)
    sgm = meta['~snsGeomMap']

    prb_params = probe_params(meta['imDatPrb_pn'])

    # get the entry for each channel
    sgm_split = re.findall(r'\((.*?)\)', sgm)

    # pull out the 
    gmap = np.zeros((len(sgm_split)-1, 4))
    for i, s in enumerate(sgm_split[1:]):
        gmap[i] = [float(x) for x in s.split(':')]
    
    gmap[:, 1] += gmap[:, 0] * prb_params['shank_spacing']

    g_df = pd.DataFrame(gmap, columns=['shank', 'x', 'y', 'used'])
    return g_df

def get_chanmap(bin_path):
    """
    Get the channel map of the binary file.
    
    Parameters
    ----------
    bin_path : str
        Path to the binary file
        
    Returns
    -------
    c_df : pd.DataFrame
        Channel map of the binary file. Each row is a channel.
        The columns are: name, order
    """

    meta = read_meta(bin_path)
    chanmap = meta['~snsChanMap']

    scm_split = re.findall(r'\((.*?)\)', chanmap)
    ch_names = []
    ch_order = []
    for s in scm_split[1:]:
        ch_names.append(s.split(':')[0])
        ch_order.append(int(s.split(':')[1]))
    
    c_df = pd.DataFrame({'name': ch_names, 'order': ch_order})
    return c_df


def probe_params(probe_pn):
    """
    Returns the probe parameters for a given probe part number
    
    Parameters
    ----------
    probe_pn : str
        Probe part number, derived from imDatPrb field in .meta file
    
    Returns
    -------
    params : dict
        Dictionary of probe parameters
    """
    
    params = {}
    if probe_pn == 'NP2014':
        params['shank_num'] = 4
        params['shank_spacing'] = 250
    else:
        raise ValueError('Probe part number not recognized')
    
    return params


def dclut_from_meta(bin_path, dcl_path=None):
    """
    Create a dclut json file from the .meta file associated with a SpikeGLX. bin file

    Parameters
    ----------
    bin_path : str
        Path to the binary file
    
    Optional
    --------
    dcl_path : str
        Path to save the dclut json file. If not provided, the file will be saved in
        the same directory as the binary file with the same name but with a .dclut extension.
    
    Returns
    -------
    dcl_path : str
        Path to the dclut json file
    """

    gmap = get_geommap(bin_path)
    chmap = get_chanmap(bin_path)
    meta = read_meta(bin_path)

    chan_props = chmap.merge(gmap, left_index=True, right_index=True, how='outer')
    chan_num = meta['nSavedChans']
    t_num = meta['fileSizeBytes'] // (2 * chan_num)

    scales = [{'name': 'time', 'dim': 0, 'unit': 'seconds', 'type': 'linear', 'val': [1/meta['imSampRate'], 0]}, 
              {'name': 'channel', 'dim': 1, 'unit': 'none', 'type': 'index', 'val': None}, 
              {'name': 'ch_name', 'dim': 1, 'unit': 'none', 'type': 'list', 'val': chan_props['name'].values}, 
              {'name': 'ch_order', 'dim': 1, 'unit': 'none', 'type': 'list', 'val': chan_props['order'].values}, 
              {'name': 'ch_x', 'dim': 1, 'unit': 'um', 'type': 'list', 'val': chan_props['x'].values}, 
              {'name': 'ch_y', 'dim': 1, 'unit': 'um', 'type': 'list', 'val': chan_props['y'].values}, 
              {'name': 'ch_shank', 'dim': 1, 'unit': 'none', 'type': 'list', 'val': chan_props['shank'].values}]
    
    # scale_names = ['time', 'channel', 'ch_name', 'ch_order', 'ch_x', 'ch_y', 'ch_shank']
    # scale_dims = [0, 1, 1, 1, 1, 1, 1]
    # scale_units = ['seconds', 'none', 'none', 'none', 'um', 'um', 'none']
    # scale_types = ['linear', 'index', 'list', 'list', 'list', 'list', 'list']
    # scale_vals = [[1/meta['imSampRate'], 0], None, chan_props['name'].values, 
    #             chan_props['order'].values, chan_props['x'].values, 
    #             chan_props['y'].values, chan_props['shank'].values]
    

    dcl_path = create_dclut(bin_path, [t_num, chan_num], dtype='int16', data_name='data',
                        data_unit='au', scales = scales)