"""Load image files
Author: Drew B. Headley
"""

import os
import glob
from PIL import Image
import numpy as np
import xarray as xr
from typing import List

def load_images_xr(directory_path: str, image_extensions: List[str] = None, 
                   insert_blank: bool = False) -> xr.DataArray:
    """Read all images from a directory into an xarray.DataArray.

    This function scans a specified directory for image files, reads them
    using the PIL (Pillow) library, and compiles them into a single
    xarray.DataArray.

    The data is organized with dimensions ('image', 'y', 'x') for
    grayscale images or ('image', 'y', 'x', 'channel') for color images.
    The 'image' dimension is indexed numerically and includes a coordinate
    'image_name' with the corresponding filenames.

    Parameters
    ----------
    directory_path : str
        The full path to the directory containing the images.
    image_extensions : list of str, optional
        A list of image file extensions to look for (e.g., ['.png', '.jpg']).
        If None, defaults to common image formats.
    insert_blank : bool, optional
        If True, insert a blank image at the beginning with image name 'blank'.

    Returns
    -------
    imgs_xr: xr.DataArray
        An xarray.DataArray containing the image data. The dimensions will
        be ('image', 'y', 'x') for grayscale or 
        ('image', 'y', 'x', 'channel') for color images.

    Raises
    ------
    FileNotFoundError
        If no images with the specified extensions are found in the directory.
    ValueError
        If the images in the directory result in an unexpected number of
        array dimensions after stacking.
    """
    if image_extensions is None:
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif']

    # Find all image files in the directory
    search_paths = [os.path.join(directory_path, f"*{ext}") for ext in image_extensions]
    image_files = []
    for path in search_paths:
        image_files.extend(glob.glob(path))

    if not image_files:
        raise FileNotFoundError(f"No images found in directory: {directory_path}")

    # Read all images into a list of numpy arrays using PIL
    images_list = [np.array(Image.open(f)) for f in image_files]
    image_names = [os.path.basename(f) for f in image_files]

    # Insert blank image if requested
    if insert_blank:
        # Use the shape and dtype of the first image
        blank_shape = images_list[0].shape
        blank_dtype = images_list[0].dtype
        blank_image = np.zeros(blank_shape, dtype=blank_dtype)
        images_list.insert(0, blank_image)
        image_names.insert(0, 'blank')

    # Stack the images into a single numpy array
    all_images_data = np.stack(images_list, axis=0)
    
    # Define dimensions based on the shape of the stacked numpy array
    if all_images_data.ndim == 3: # Grayscale images
        dims = ("image", "y", "x")
        coords = {
            "image": np.arange(len(image_names)),
            "y": np.arange(all_images_data.shape[1]),
            "x": np.arange(all_images_data.shape[2]),
            "image_name": ("image", image_names),
        }
    elif all_images_data.ndim == 4: # Color images
        dims = ("image", "y", "x", "channel")
        coords = {
            "image": np.arange(len(image_names)),
            "y": np.arange(all_images_data.shape[1]),
            "x": np.arange(all_images_data.shape[2]),
            "channel": np.arange(all_images_data.shape[3]),
            "image_name": ("image", image_names),
        }
    else:
        raise ValueError(f"Unexpected number of dimensions in image data: {all_images_data.ndim}")

    # Create the xarray DataArray
    imgs_xr = xr.DataArray(
        all_images_data,
        dims=dims,
        coords=coords,
    )
    
    # Add attributes for metadata
    imgs_xr.attrs['directory_path'] = directory_path
    
    return imgs_xr

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    try:
        # Create a dummy directory and some dummy images for demonstration
        img_path = os.path.join('..', '..', 'test_data', 'invitro', 'single_squares')
        
        # Convert the images in the directory to an xarray DataArray
        imgs_xr = load_images_xr(img_path)

        # Print the resulting xarray DataArray to see its structure
        print("Successfully converted image directory to xarray.DataArray:")
        print(imgs_xr)

        # Plot the selected image
        imgs_xr.mean(dim='image').plot(cmap='gray')
        plt.title("Average image")
        plt.show()

    except Exception as e:
        print(f"An error occurred. Please ensure you have a directory with images.")
        print(f"Error details: {e}")