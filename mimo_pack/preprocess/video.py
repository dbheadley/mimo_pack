# Video preprocessing functions
# Author: Drew Headley (with github copilot assistance)
# Date: 2024-06-10

import cv2
import os
import glob
import re
from tqdm import tqdm
import pandas as pd
import numpy as np

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

    Returns
    -------
    vid_path : str
        Path to the output video file.
    tbl_path : str
        Path to the output table file.
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
                tbl_df = pd.read_csv(f, index_col=0)

            # add or overwrite the ROI columns in to the existing table
            for col in roi_df.columns:
                tbl_df[col] = roi_df[col]

            tbl_df.to_csv(table_path, index=True)

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

def extract_rois_to_videos(video_path, display_frame=0, output_dir=None):
    """
    Allows user to select multiple ROIs on a given frame, then saves a new video for each ROI.
    User can adjust brightness and contrast before ROI selection. Videos are saved with the adjusted settings.
    
    Parameters
    ----------
    video_path : str
        Path to the video file.
    display_frame : int, optional
        Frame index to display for ROI selection (default: 0).
    output_dir : str, optional
        Directory to save ROI videos. If None, saves in the same directory as video_path.

    Returns
    -------
    roi_videos : list of str
        List of output video file paths, one per ROI.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    # Go to the display frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, display_frame)
    ret, frame = cap.read()
    if not ret:
        raise IOError(f"Error reading frame {display_frame} from video file: {video_path}")

    # Default values for contrast and brightness
    contrast = 100  # 100 means 1.0
    brightness = 100  # 100 means 0

    # Window for ROI selection
    window_name = 'Select ROIs'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Trackbar callback
    def update_display(_=None):
        alpha = cv2.getTrackbarPos('Contrast', window_name) / 100.0  # 0.0 - 3.0
        beta = cv2.getTrackbarPos('Brightness', window_name) - 100   # -100 to 100
        adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        cv2.imshow(window_name, adjusted)

    # Create trackbars
    cv2.createTrackbar('Contrast', window_name, contrast, 300, update_display)
    cv2.createTrackbar('Brightness', window_name, brightness, 200, update_display)
    update_display()

    # Wait for user to press any key to continue
    cv2.waitKey(0)
    # Get final contrast/brightness values
    final_contrast = cv2.getTrackbarPos('Contrast', window_name) / 100.0
    final_brightness = cv2.getTrackbarPos('Brightness', window_name) - 100

    # Show adjusted frame for ROI selection
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=final_contrast, beta=final_brightness)
    cv2.imshow(window_name, adjusted_frame)
    rois = cv2.selectROIs(window_name, adjusted_frame, fromCenter=False)
    cv2.destroyWindow(window_name)

    if len(rois) == 0:
        print("No ROIs selected.")
        return []

    # Prepare output directory
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    os.makedirs(output_dir, exist_ok=True)
    file_root = os.path.splitext(os.path.basename(video_path))[0]

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare video writers for each ROI
    writers = []
    out_paths = []
    for idx, (x, y, w, h) in enumerate(rois):
        out_path = os.path.join(output_dir, f'{file_root}_roi_{idx+1}.mp4')
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (w, h),
            isColor=True
        )
        writers.append(writer)
        out_paths.append(out_path)

    # Reset to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # Apply contrast and brightness to each frame
        adj_frame = cv2.convertScaleAbs(frame, alpha=final_contrast, beta=final_brightness)
        for idx, (x, y, w, h) in enumerate(rois):
            roi_frame = adj_frame[y:y+h, x:x+w]
            writers[idx].write(roi_frame)

    # Release everything
    cap.release()
    for writer in writers:
        writer.release()

    print(f"Saved {len(out_paths)} ROI video(s):")
    for path in out_paths:
        print(path)

    return out_paths