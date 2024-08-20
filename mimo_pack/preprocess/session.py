# Functions for preprocessing entire sessions
# Author: Drew Headley, 2024

import os
import pandas as pd
from mimo_pack.util.files import find_all_matching_files
from mimo_pack.preprocess.lfp import make_lfp_file_spikeglx, dclut_from_meta_lfp
from mimo_pack.preprocess.time import align_sync_dclut
from mimo_pack.fileio.spikeglx import dclut_from_meta


def preprocess_spikeglx(sess_dir, sync_ap={'channel': [384]}, sync_nidq={'channel': [5]}, use_catgt_version=False):
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

    # make a LFP file from the imec AP files
    lfp_files = []
    lfp_dclut_files = []
    # make a LFP file from the imec files
    for ap_file in ap_files:
        print('Processing {}'.format(os.path.basename(ap_file)))
        lfp_files.append(make_lfp_file_spikeglx(ap_file, verbose=True))
        lfp_dclut_files.append(dclut_from_meta_lfp(lfp_files[-1]))

    # create dclut json files
    ap_dclut_files = [dclut_from_meta(f) for f in ap_files]
    nidq_dclut_file = dclut_from_meta(nidq_file)

    if use_catgt_version:
        lfp_dclut_files = find_all_matching_files(sess_dir, r'tcat\.imec([0-9]+)\.lfp_dclut\.json')
    else:
        lfp_dclut_files = find_all_matching_files(sess_dir, r't0\.imec([0-9]+)\.lfp_dclut\.json')

    # align the times across all the dclut files to the first ap file
    ap_r = ap_dclut_files[0] 
    sync_scale_name = 'time'
    for ap_t in range(1, len(ap_dclut_files)):
        file_t = ap_dclut_files[ap_t]
        print('Aligning {} to {}'.format(os.path.basename(file_t), os.path.basename(ap_r)))
        align_sync_dclut(file_t, ap_r, sync_ap, sync_ap, sync_scale_name, verbose=True)

    # Align the LFP files to the first AP file
    for lfp_t in range(len(lfp_dclut_files)):
        file_t = lfp_dclut_files[lfp_t]
        print('Aligning {} to {}'.format(os.path.basename(file_t), os.path.basename(ap_r)))
        align_sync_dclut(file_t, ap_r, sync_ap, sync_ap, sync_scale_name, verbose=True)

    # Align the NIDQ file to the first AP file
    print('Aligning {} to {}'.format(os.path.basename(nidq_dclut_file), os.path.basename(ap_r)))
    align_sync_dclut(nidq_dclut_file, ap_r, sync_nidq, sync_ap, sync_scale_name, verbose=True)

    # create a dataframe with the file names, first column is the session dir, second is the probe directory, 
    # third is the file, fourth is its dclut json file, and last is the type of file
    files = []
    for f in range(len(ap_files)):
        files.append([sess_dir, ap_files[f], ap_dclut_files[f], 'ap'])
    for f in range(len(lfp_files)):
        files.append([sess_dir,lfp_files[f], lfp_dclut_files[f], 'lfp'])
    files.append([sess_dir, nidq_file, nidq_dclut_file, 'nidq'])

    df = pd.DataFrame(files, columns=['session_dir', 'file', 'dclut_file', 'type'])
    return df

