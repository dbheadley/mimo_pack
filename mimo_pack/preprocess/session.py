# Functions for preprocessing entire sessions
# Author: Drew Headley, 2024

import os
import pandas as pd
from mimo_pack.util.files import find_all_matching_files
from mimo_pack.preprocess.lfp import make_lfp_file_dclut
from mimo_pack.preprocess.time import align_sync_dclut
from mimo_pack.fileio.spikeglx import dclut_from_meta


def preprocess_spikeglx(sess_dir, sync_ap={'channel': [384]}, sync_nidq={'channel': [5]}, 
                        use_catgt_version=False):
    """
    Takes a session directory and preprocesses all the imec files in the directory. This 
    includes creating LFP files from the AP files, creating dclut json files for all the 
    files, and aligning the times across all the files to the first AP file. The function 
    returns a dataframe with the file names, the dclut json files, and the type of file.

    Parameters
    ----------
    sess_dir : str
        The session directory

    Optional
    --------
    sync_ap : dict
        The sync channel for the AP files as specified for dclut. 
        Default is {'channel': [384]}
    sync_nidq : dict
        The sync channel for the NIDQ file as specified for dclut. 
        Default is {'channel': [5]}
    use_catgt_version : bool
        Whether to use the CatGT version of the files. 
        Default is False

    Returns
    -------
    df : pd.DataFrame
        A dataframe with the file names, the dclut json files, and the type of file
    """
    
    # identify all ap and nidq files in the session directory
    if use_catgt_version:
        ap_files = find_all_matching_files(sess_dir, r'tcat\.imec([0-9]+)\.ap\.bin')
    else:
        ap_files = find_all_matching_files(sess_dir, r't0\.imec([0-9]+)\.ap\.bin')
    nidq_file = find_all_matching_files(sess_dir, r't0\.nidq\.bin')[0] # there should be only one nidq file

    print("ap files:")
    for f in ap_files:
        print(f)

    print("nidq file:")
    print(nidq_file)

    # create dclut json files
    ap_dclut_files = [dclut_from_meta(f) for f in ap_files]
    nidq_dclut_file = dclut_from_meta(nidq_file)

    # align the times across all the dclut files to the first ap file
    ap_r = ap_dclut_files[0] 
    sync_scale_name = 'time'
    for ap_t in range(1, len(ap_dclut_files)):
        file_t = ap_dclut_files[ap_t]
        print('Aligning {} to {}'.format(os.path.basename(file_t), os.path.basename(ap_r)))
        align_sync_dclut(file_t, ap_r, sync_ap, sync_ap, sync_scale_name, verbose=True)
    
    # Align the NIDQ file to the first AP file
    print('Aligning {} to {}'.format(os.path.basename(nidq_dclut_file), os.path.basename(ap_r)))
    align_sync_dclut(nidq_dclut_file, ap_r, sync_nidq, sync_ap, sync_scale_name, verbose=True)

    # make a LFP file from the imec AP files with dclut 
    lfp_files = []
    lfp_dclut_files = []
    # make a LFP file from the imec files 
    for file in ap_dclut_files:
        print('Processing {}'.format(os.path.basename(file)))
        lfp_files.append(file.replace('ap_dclut.json','lfp.bin'))
        lfp_dclut_files.append(make_lfp_file_dclut(file, lfp_files[-1], verbose=True))

    # create a dataframe with the file names, first column is the session dir, second is the probe directory, 
    # third is the file, fourth is its dclut json file, and last is the type of file
    files = []
    for f in range(len(ap_files)):
        files.append([sess_dir, ap_files[f], ap_dclut_files[f], 'ap'])
    for f in range(len(lfp_files)):
        files.append([sess_dir, lfp_files[f], lfp_dclut_files[f], 'lfp'])
    files.append([sess_dir, nidq_file, nidq_dclut_file, 'nidq'])

    df = pd.DataFrame(files, columns=['session_dir', 'file', 'dclut_file', 'type'])
    return df