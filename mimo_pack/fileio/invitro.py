"""Organize in vitro data
Author: Drew B. Headley
"""

import os
import pandas as pd
from mimo_pack.util.files import find_all_matching_files, split_file_path
from .abf import load_abf_settings

def invitro_files_df(rootdir: str, required_ext='.abf') -> pd.DataFrame:
    """
    Scans a directory of in vitro experimental files, extracts metadata from their paths, 
    and returns a pivoted DataFrame.

    Parameters
    ----------
    rootdir : str
        The root directory path to search for files.
    required_ext : str, list of str, or None, optional
        File extension(s) that must be present for a row to be kept in the final dataframe.
        If None, no extension is required. Default is '.abf'.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by 'subject', 'slice', and 'cell', with columns for each file extension.
    
    Notes
    -----
    """

    # Find all files matching the pattern in the specified directory
    files = find_all_matching_files(rootdir, '.*')
    files = pd.DataFrame(files, columns=['file_path'])

    # Create a column for the last, second to last, and third to last directory names.
    files['cell'] = files['file_path'].apply(lambda x: split_file_path(x)[-2])
    files['slice'] = files['file_path'].apply(lambda x: split_file_path(x)[-3])
    files['subject'] = files['file_path'].apply(lambda x: split_file_path(x)[-4])

    # Add column for file name extension
    files['ext'] = files['file_path'].apply(lambda x: os.path.splitext(x)[1])

    # Pivot the dataframe to have one row per cell, slice, and subject
    # and a column for each type of file path
    files = files.pivot_table(index=['subject', 'slice', 'cell'],
                              values='file_path',
                              columns='ext',
                              aggfunc=list).reset_index()

    # Handle required_ext argument
    if required_ext is None:
        pass  # No filtering
    else:
        if isinstance(required_ext, str):
            required_ext = [required_ext]
       
        # Keep only rows where all required extensions are present (not NaN)
        mask = files[required_ext].notna().all(axis=1)
        files = files[mask]

    # if .abf column is present, create a column with the settings
    # of each abf file
    if '.abf' in files.columns:
        files['protocols'] = files['.abf'].apply(lambda x: load_abf_settings_dict(x) if x else {})
    else:
        files['protocols'] = None
        
    return files

# function that accepts a list of abf files, loads the settings for each file, then return
# a dictionary where the keys are the protocols and the value are the lists of files
def load_abf_settings_dict(abf_files: list) -> dict:
    
    """
    Organizes ABF files by protocol type.

    Given a list of ABF file paths, this function loads the settings for each file,
    extracts the protocol name, and groups the files into a dictionary where each key
    is a protocol and the value is a list of ABF files using that protocol.

    Parameters
    ----------
    abf_files : list of str
        List of ABF file paths.

    Returns
    -------
    dict
        Dictionary mapping protocol names (str) to lists of ABF file paths (list of str).
    """
    settings_dict = {}
    for abf_file in abf_files:
        settings = load_abf_settings(abf_file)
        protocol = settings.get('protocol', 'unknown')
        if protocol not in settings_dict:
            settings_dict[protocol] = []
        settings_dict[protocol].append(abf_file)
    return settings_dict