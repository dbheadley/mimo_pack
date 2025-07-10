"""Load image files
Author: Drew B. Headley
"""
import os
import glob
from PIL import Image
import numpy as np
import xarray as xr
from typing import List

def image_dir_to_xarray(directory_path: str, image_extensions: List[str] = None) -> xr.DataArray:
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

    Returns
    -------
    xr.DataArray
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
    
    # Get the filenames to use as a coordinate
    image_names = [os.path.basename(f) for f in image_files]

    # Stack the images into a single numpy array
    # This will create a 3D array for grayscale or 4D for color images
    all_images_data = np.stack(images_list, axis=0)
    
    # Define dimensions based on the shape of the stacked numpy array
    if all_images_data.ndim == 3: # Grayscale images
        dims = ("image", "y", "x")
        coords = {
            "image": np.arange(len(image_files)),
            "y": np.arange(all_images_data.shape[1]),
            "x": np.arange(all_images_data.shape[2]),
            "image_name": ("image", image_names),
        }
    elif all_images_data.ndim == 4: # Color images
        dims = ("image", "y", "x", "channel")
        coords = {
            "image": np.arange(len(image_files)),
            "y": np.arange(all_images_data.shape[1]),
            "x": np.arange(all_images_data.shape[2]),
            "channel": np.arange(all_images_data.shape[3]),
            "image_name": ("image", image_names),
        }
    else:
        raise ValueError(f"Unexpected number of dimensions in image data: {all_images_data.ndim}")


    # Create the xarray DataArray
    xarray_images = xr.DataArray(
        all_images_data,
        dims=dims,
        coords=coords,
    )
    
    # Add attributes for metadata
    xarray_images.attrs['directory_path'] = directory_path
    
    return xarray_images

if __name__ == '__main__':
    # This is an example of how to use the function.
    # You will need to create a directory with some images to test this.
    
    try:
        # Create a dummy directory and some dummy images for demonstration
        temp_dir = 'temp_image_dir'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Create two dummy grayscale images
        img1_data = np.random.randint(0, 256, size=(100, 80), dtype=np.uint8)
        img2_data = np.random.randint(0, 256, size=(100, 80), dtype=np.uint8)
        
        # Save the dummy images using PIL
        Image.fromarray(img1_data).save(os.path.join(temp_dir, 'test_image_1.png'))
        Image.fromarray(img2_data).save(os.path.join(temp_dir, 'test_image_2.png'))

        # Specify the path to the directory
        image_directory = temp_dir
        
        # Convert the images in the directory to an xarray DataArray
        xarray_of_images = image_dir_to_xarray(image_directory)

        # Print the resulting xarray DataArray to see its structure
        print("Successfully converted image directory to xarray.DataArray:")
        print(xarray_of_images)

        # You can now easily perform operations on the data, for example:
        # Select an image by its name
        first_image = xarray_of_images.sel(image_name='test_image_1.png')
        
        print("\nData for 'test_image_1.png':")
        print(first_image)

        # Plot the selected image
        import matplotlib.pyplot as plt
        first_image.plot(cmap='gray')
        plt.title("First Image from Xarray")
        plt.show()

    except Exception as e:
        print(f"An error occurred. Please ensure you have a directory with images.")
        print(f"Error details: {e}")
    finally:
        # Clean up the dummy files and directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)