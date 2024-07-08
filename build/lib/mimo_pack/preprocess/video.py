""" Video preprocessing functions
    Author: Drew Headley (with github copilot assistance)
    Date: 2024-06-10
    """

import cv2
import os
import glob
import argparse
import re
from tqdm import tqdm
import pandas as pd


def tif_to_video(folder_path, output_filename, fps=30):
    """ Convert TIF images to a video. 
    Usage: python tif_to_video.py <folder_path> <output_filename> [--fps <fps>]
    folder_path: Path to the folder containing TIF images.
    output_filename: Name of the output video file.
    fps: Frames per second for the output video. Default is 30.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing TIF images.
    output_filename : str
        Name of the output video file.
    fps : int, optional
        Frames per second for the output video. Default is 30.
    """

    # Get a list of all TIF files in the folder
    tif_files = glob.glob(os.path.join(folder_path, '*.tif'))
    tif_files.sort()
    
    # for each file name, get the number between the second to last and last underscore
    # this is the epoch number
    epoch_nums = [int(re.search(r'_(\d+)_(\d+).tif', f).group(1)) for f in tif_files]
    frame_nums = [int(re.search(r'_(\d+).tif', f).group(1)) for f in tif_files]

    files_df = pd.DataFrame({'tif_file': tif_files, 'epoch_num': epoch_nums, 'frame_num': frame_nums})
    files_df.sort_values(['epoch_num', 'frame_num'], inplace=True)

    img = cv2.imread(tif_files[0])
    height, width, _ = img.shape

    # Create a VideoWriter object
    video = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'I420'), fps, (width, height))

    # Loop over all TIF files
    for curr_row in tqdm(files_df.iterrows(), total=len(files_df)):
        img = cv2.imread(curr_row[1]['tif_file'])
        video.write(img)

    # Close the VideoWriter object
    video.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert TIF images to a video.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing TIF images.')
    parser.add_argument('output_filename', type=str, help='Name of the output video file.')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the output video.')

    args = parser.parse_args()
    print('Converting TIF images to video...')
    print('Folder path:', args.folder_path)
    print('Output filename:', args.output_filename)
    print('FPS:', args.fps)
    tif_to_video(args.folder_path, args.output_filename, args.fps)