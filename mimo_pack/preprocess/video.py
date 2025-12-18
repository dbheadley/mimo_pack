# Video preprocessing functions
# Author: Drew Headley (with github copilot assistance)
# Date: 2024-06-10

import cv2
import os
import glob
import sys
from contextlib import redirect_stdout
import argparse
import re
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import pdb

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from mimo_pack.preprocess.time import align_sync_video_barcode
from mimo_pack.plot.video import plot_video_data

def tif_to_video(folder_path, output_base=None, fps=30):
    """ Convert TIF images to a video. 
    Usage: python tif_to_video.py <folder_path> <output_filename> [--fps <fps>]
    folder_path: Path to the folder containing TIF images.
    output_base: Base name of the output files.
    fps: Frames per second for the output video. Default is 30.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing TIF images.
    output_base : str
        Base name of the output files.
    fps : int, optional
        Frames per second for the output video. Default is 30.
    """

    # Get a list of all TIF files in the folder
    tif_paths = glob.glob(os.path.join(folder_path, '*.tif'))
    
    # get just file name from each file path
    tif_files = [os.path.basename(f) for f in tif_paths]

    if output_base is None:
        # if no output base is provided, create one using first tif file name
        output_base = re.search(r'(.+[a-zA-Z])\d+_\d+.tif', tif_files[0]).group(1)
    
    vid_path = os.path.join(folder_path, output_base + '.mp4')
    tbl_path = os.path.join(folder_path, output_base + '.csv')

    # for each file name, get the number between the second to last and last underscore
    # this is the epoch number
    matches = [re.search(r'(\d+)_(\d+).tif', f) for f in tif_files]
    epoch_nums = [int(m.group(1)) for m in matches]
    frame_nums = [int(m.group(2)) for m in matches]

    files_df = pd.DataFrame({'tif_path': tif_paths, 'tif_file': tif_files, 
                             'epoch_num': epoch_nums, 'frame_num': frame_nums})
    files_df.sort_values(['epoch_num', 'frame_num'], inplace=True)
    files_df.set_index(pd.Index(np.arange(len(files_df)), name='frame'), 
                       inplace=True)

    img = cv2.imread(tif_paths[0])
    height, width, _ = img.shape

    # Create a VideoWriter object
    video = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'h264'), 
                            fps, (width, height), isColor=True)

    # Loop over all TIF files
    for curr_row in tqdm(files_df.iterrows(), total=len(files_df)):
        img = cv2.imread(curr_row[1]['tif_path'])
        #pdb.set_trace()
        video.write(img)

    # Close the VideoWriter object
    video.release()
    
    # write frame data to csv file
    files_df = files_df.drop(columns=['tif_path'])
    files_df.to_csv(tbl_path)
    return vid_path, tbl_path


def video_frame_table(video_path: str, out_csv: str | None = None) -> pd.DataFrame:
    """
    Build and (optionally) save a per-frame table for a video.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    out_csv : str or None
        If provided, save the table to this CSV path. If None, save next to the video with suffix '_frames.csv'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['video', 'epoch', 'frame'] where epoch is 0 and each row corresponds to a frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    basename = os.path.basename(video_path)
    df = pd.DataFrame({
        'video': [basename] * frame_count,
        'epoch': [0] * frame_count,
        'frame': np.arange(frame_count, dtype=int)
    })

    if out_csv is None:
        base, _ = os.path.splitext(basename)
        out_csv = os.path.join(os.path.dirname(video_path), f"{base}_frames.csv")

    df.to_csv(out_csv, index=False)
    return df


def create_video_rois(video_path, frame_time=0):
    """ Display a video frame and allow the user to select multiple ROIs.
    The selected ROIs are saved to a json file.
    
    Parameters
    ----------
    video_path : str
        Path to the video file.
    frame_time : int, optional
        Time in seconds for frame to be displayed. Default is 0 (first frame).
    
    Returns
    -------
    rois : list of tuples
        List of selected ROIs, each ROI is a tuple of (x, y, width, height).
    """

    # open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")
    
    # get the frame for the specified time
    if frame_time > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = cap.read()
    if not ret:
        raise IOError(f"Error reading frame from video file: {video_path}")
    
    # create a window to display the video
    cv2.namedWindow('Select ROIs', cv2.WINDOW_NORMAL)
    
    # resize window to have same aspect ratio as the video
    height, width, _ = frame.shape
    aspect_ratio = width / height
    new_width = 800
    new_height = int(new_width / aspect_ratio)
    cv2.resizeWindow('Select ROIs', new_width, new_height)
    cv2.imshow('Select ROIs', frame)

    adjusted_frame = frame.copy()  # Create a copy of the original frame to store adjustments

    def adjust_brightness(val):
        """Callback function to adjust brightness."""
        alpha = val / 100  # Scale brightness (0 to 2)
        nonlocal adjusted_frame  # Use the adjusted frame for cumulative changes
        adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
        cv2.imshow('Select ROIs', adjusted_frame)

    # Create a trackbar for brightness adjustment
    cv2.createTrackbar('Brightness', 'Select ROIs', 100, 200, adjust_brightness)
    adjust_brightness(100)  # Initialize with default brightness

    cv2.waitKey(0)  # Wait for a key press

    # allow user to select multiple ROIs
    rois = cv2.selectROIs('Select ROIs', adjusted_frame, fromCenter=False)
    
    # close the window
    cv2.destroyWindow('Select ROIs')
    
    # convert ROIs to a list of tuples
    rois = [(int(x), int(y), int(w), int(h)) for x, y, w, h in rois]

    # Create a directory for quality control reports
    qc_dir = os.path.join(os.path.dirname(video_path), 'qc_reports')
    os.makedirs(qc_dir, exist_ok=True)

    # Create a figure and plot the frame with ROIs
    fig, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2RGB))
    
    for i, (x, y, w, h) in enumerate(rois):
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 10, f'ROI {i}', color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.5, pad=0))

    ax.set_title('Selected ROIs')
    ax.axis('off')
    
    # Save the figure
    output_pdf_path = os.path.join(qc_dir, 'selected_rois.pdf')
    fig.savefig(output_pdf_path)

    return rois

    
def track_video_rois(video_path, table_path=None, rois=None, roi_suffix='roi_'):
    """ Track mean value inthe selected ROIs
    
    Parameters
    ----------
    video_path : str
        Path to the video file.
    table_path : str
        Path to the frame data table. ROIs are added to this table. 
        If None, no table file is used.
    rois : list of tuples
        List of selected ROIs, each ROI is a tuple of (x, y, width, height).
    roi_suffix : str
        Suffix for the ROI columns in the output DataFrame.
        
    Returns
    -------
    roi_df : dataframe
        DataFrame containing the mean values for each ROI over time. Indexed 
        by frame number.
    """
        
    # open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")
    
    # get the number of frames in the video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    roi_arr = np.zeros((num_frames, len(rois), 3))

    for i in tqdm(range(num_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # loop over the ROIs
        for r, roi in enumerate(rois):
            x, y, w, h = roi
            # extract the ROI from the frame
            roi_frame = frame[y:y+h, x:x+w]
            # calculate the mean value of the ROI
            roi_arr[i, r, :] = cv2.mean(roi_frame)[:3]
    
    # get the mean value of the ROIs across the color channels
    roi_arr = np.mean(roi_arr, axis=2)

    # convert the ROI array to a DataFrame
    roi_df = pd.DataFrame(roi_arr, columns=[roi_suffix + f'{i}' for i in 
                                            range(len(rois))])
    roi_df.set_index(pd.Index(range(roi_df.shape[0]), name='frame'), inplace=True)

    # save the ROI data to a CSV file if a table path is provided
    # when the table file already exists, the new data is appended to it
    if table_path is not None:
        if not os.path.exists(table_path):
            roi_df.to_csv(table_path, index=True)
        else:
            with open(table_path, 'r') as f:
                tbl_df = pd.read_csv(f)
            tbl_df.set_index('frame', inplace=True)
            # add or overwrite the ROI columns in to the existing table
            for col in roi_df.columns:
                tbl_df[col] = roi_df[col]

            tbl_df.to_csv(table_path, index=True)

    return roi_df


def track_video_diff(video_path, table_path=None, rois=None, 
                     roi_suffix='roi_', lag_seconds=1, diff_thresh=None):
    """
    Track mean absolute difference in selected ROIs between frames separated by a time lag.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    table_path : str or None
        Path to a CSV table to update/append the diff columns. If None, no file I/O.
    rois : list of tuples or None
        List of ROIs as (x, y, w, h). If None, the whole frame is used as one ROI.
    roi_suffix : str
        Prefix for output column names.
    lag_seconds : float
        Time lag (in seconds) between frames used to compute the difference. Default ~1/30 s.
    diff_thresh : float or None
        If provided, values below this threshold are set to zero. Default is None (no thresholding).

    Returns
    -------
    roi_df : pandas.DataFrame
        DataFrame with mean absolute differences per ROI for each frame index. Frames
        that don't have a future frame separated by lag_seconds remain NaN.
    """
    from collections import deque

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    lag_frames = max(1, int(round(lag_seconds * fps)))

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Error if lag exceeds or equals video duration
    if lag_frames >= num_frames:
        cap.release()
        raise ValueError(f"lag ({lag_frames} frames) >= total number of frames ({num_frames}). Reduce lag_seconds.")

    # Prepare result array filled with NaN
    n_rois = len(rois) if rois is not None else 1
    roi_arr = np.full((num_frames, n_rois), np.nan, dtype=float)

    # If no rois provided, we will determine full-frame size from the first read frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Read and buffer the first lag_frames frames (we will not read all frames at once)
    buf = deque(maxlen=lag_frames)
    for i in range(lag_frames):
        ret, frm = cap.read()
        if not ret:
            cap.release()
            raise IOError(f"Unable to read initial frame {i} while filling buffer.")
        buf.append(frm.astype(np.float32))

    # Determine ROIs if not provided using the first buffered frame
    height, width = buf[0].shape[:2]
    if rois is None:
        rois = [(0, 0, width, height)]

    # Now iterate reading one future frame at a time and compute difference with the oldest buffered frame
    for idx in tqdm(range(num_frames - lag_frames)):
        # read the future frame (frame index = idx + lag_frames)
        ret, fut = cap.read()
        if not ret:
            break

        curr = buf[0]  # oldest buffered frame corresponds to index idx

        # compute per-ROI mean absolute difference averaged across pixels and channels
        for r, (x, y, w, h) in enumerate(rois):
            curr_roi = curr[y:y+h, x:x+w]
            fut_roi = fut[y:y+h, x:x+w]
            diff_roi = np.abs(fut_roi - curr_roi)
            if diff_thresh is not None:
                diff_roi[diff_roi < diff_thresh] = 0
            val = np.mean(diff_roi)
            roi_arr[idx+lag_frames, r] = val

        # update buffer with the newly read future frame
        buf.append(fut.astype(np.float32))

    cap.release()

    # Build DataFrame
    colnames = [f"{roi_suffix}{i}_diff" for i in range(n_rois)]
    roi_df = pd.DataFrame(roi_arr, columns=colnames)
    roi_df.index = pd.Index(range(num_frames), name='frame')

    # save/append to table if requested
    if table_path is not None:
        if not os.path.exists(table_path):
            roi_df.to_csv(table_path)
        else:
            tbl_df = pd.read_csv(table_path)
            tbl_df.set_index('frame', inplace=True)
            for col in roi_df.columns:
                tbl_df[col] = roi_df[col]
            tbl_df.to_csv(table_path)

    return roi_df


def rois_to_barcodes(table_path, roi_cols, roi_thresh=200, barcode_name='barcode'):
    """ Convert the values of the ROIs on each frame to a barcode.
    Update the table file with the barcode values. Each barcode is
    represented as an integer value.

    Parameters
    ----------
    table_path : str
        Path to the table file.
    roi_cols : list of str
        List of ROI column names.
    roi_thresh : int or ndarray
        Threshold value for the ROI values to be conidered as True.
        If an array is provided, it should have the same length as roi_cols.
    barcode_name : str
        Name of the barcode column in the output DataFrame.
        Default is 'barcode'.

    Returns
    -------
    barcode_df : DataFrame
        DataFrame containing the barcode values for each frame.
    """

    roi_thresh = np.array(roi_thresh, ndmin=1)
    if roi_thresh.size == 1:
        roi_thresh = np.full(len(roi_cols), roi_thresh[0])
    elif roi_thresh.size != len(roi_cols):
        raise ValueError(f"roi_thresh should be a scalar or an array of length {len(roi_cols)}")
    
    # read the table file
    tbl_df = pd.read_csv(table_path, index_col=0)

    # get the ROI values
    roi_vals = tbl_df[roi_cols].values

    # convert barcode value to binary
    roi_vals = roi_vals>roi_thresh

    pos_vals = np.pow(2, np.arange(roi_vals.shape[1]))

    barcodes = np.sum(roi_vals * pos_vals, axis=1).astype(int)

    # add the barcode values to the table
    tbl_df[barcode_name] = barcodes
    tbl_df.to_csv(table_path, index=True)

    barcode_df = tbl_df[[barcode_name]].copy()
    return barcode_df


def process_and_align_video(vid_path, tbl_path, rois, dcl_path, sync_channel_info):
    """Process and align video data with neural recordings.

    This function takes video data and associated metadata, tracks ROIs to
    generate barcodes, aligns these barcodes with neural recording data, and
    saves quality control (QC) plots and logs to a 'qc_reports' directory.

    Parameters
    ----------
    vid_path : str
        Path to the video file (.mp4).
    tbl_path : str
        Path to the video metadata table (.csv).
    rois : list
        List of ROIs for barcode tracking.
    dcl_path : str
        Path to the dclut file for neural data alignment.
    sync_channel_info : dict
        Dictionary specifying the sync channel.

    Returns
    -------
    str
        The path to the CSV table after alignment, which includes the aligned
        timestamps.
    """
    # Create a directory for quality control reports
    qc_dir = os.path.join(os.path.dirname(vid_path), 'qc_reports')
    os.makedirs(qc_dir, exist_ok=True)

    # Track ROIs across video frames
    roi_df = track_video_rois(vid_path, tbl_path, rois=rois)

    # Generate and save a plot of barcode signals
    roi_vals = roi_df.values
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.imshow(roi_vals[:200].T, aspect='auto', cmap='gray', interpolation='none')
    ax.set_ylabel('ROI')
    ax.set_xlabel('Frame')
    ax.set_title('Barcodes Across Frames')
    fig.savefig(os.path.join(qc_dir, 'barcode_signals.pdf'))

    # Convert ROI signals to a barcode sequence in the table
    rois_to_barcodes(tbl_path, roi_cols=['roi_' + str(i) for i in range(len(rois))])

    # Align video barcodes to neural recording times
    # Redirect verbose output to a text file
    
    log_path = os.path.join(qc_dir, 'alignment_log.txt')
    with open(log_path, 'w') as f:
        with redirect_stdout(f):
            aligned_tbl_path = align_sync_video_barcode(
                tbl_path, dcl_path, 'barcode', sync_channel_info, 'time', verbose=True
            )

    # Generate and save a plot of the aligned video data
    fig, ax = plt.subplots(figsize=(8, 2))
    plot_video_data(aligned_tbl_path, ax=ax)
    fig.savefig(os.path.join(qc_dir, 'aligned_video_data.pdf'))

    return aligned_tbl_path

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Convert TIF images to a video.')
#     parser.add_argument('folder_path', type=str, help='Path to the folder containing TIF images.')
#     parser.add_argument('output_base', type=str, help='Base name of the output files.')
#     parser.add_argument('--fps', type=int, default=30, help='Frames per second for the output video.')

#     args = parser.parse_args()
#     print('Converting TIF images to video...')
#     print('Folder path:', args.folder_path)
#     print('Output base:', args.output_base)
#     print('FPS:', args.fps)
#     tif_to_video(args.folder_path, args.output_base, args.fps)