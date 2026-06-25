# Short generic utility functions for working with hdf5 objects
# Author: Drew Headley
# Date: 2025-07-22

import h5py
import numpy as np

def get_h5_hierarchy(h5_data: h5py.File) -> list[tuple]:
    """
    Traverses an HDF5 file and returns its complete hierarchy.

    This function uses the `visititems` method from the h5py library to
    recursively visit every group and dataset in the file. It collects
    the full path and type of each item.

    Args:
        h5_data : The h5 file object to traverse.

    Returns:
        list: A list of tuples, where each tuple contains the full path
              (str) and the type (str, 'Group' or 'Dataset') of an item
              in the HDF5 file. Returns an empty list if the file
              cannot be opened or does not exist.
    """
    hierarchy = []
    
    def visit_item(name, obj):
        """
        Callback function for h5py's visititems method.
        
        Args:
            name (str): The name/path of the item.
            obj (h5py.Group or h5py.Dataset): The h5py object.
        """
        item_type = ""
        if isinstance(obj, h5py.Group):
            item_type = "Group"
            item_value = None
        elif isinstance(obj, h5py.Dataset):
            item_type = "Dataset"
            item_value = np.array(obj)
        else:
            item_type = "Unknown"
            item_value = None
            
        hierarchy.append((name, item_type, item_value))

    h5_data.visititems(visit_item)
        
    return hierarchy