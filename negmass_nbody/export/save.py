#!/usr/bin/env python

"""save.py: Functions for saving data from N-body runs."""

import h5py
import dask.array as da

__author__ = "Jamie Farnes"
__email__ = "jamie.farnes@oerc.ox.ac.uk"


def save_data(my_array, my_file_name, chunks_value):
    """Save the data to an .hdf5 file.
    
    Args:
    my_array (dask array): dask array to be saved.
    my_file_name (str): file name to save to on disk.
    chunks_value (float): dask chunks value.
    """
    x_hdf = da.from_array(my_array, chunks=(chunks_value))
    x_hdf.to_hdf5(my_file_name, '/x', compression='lzf', shuffle=True)
    return
