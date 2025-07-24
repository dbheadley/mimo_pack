"""Organize in vitro data
Author: Drew B. Headley
"""

import os
import pandas as pd
from mimo_pack.util.files import find_all_matching_files, split_file_path


def invitro_files_df(fpath: str, required_ext='.abf') -> pd.DataFrame:
    """
    Scans a directory of in vitro experimental files, extracts metadata from their paths, 
    and returns a pivoted DataFrame.

    Parameters
    ----------
    fpath : str
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
    files = find_all_matching_files(fpath, '.*')
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

    return files
