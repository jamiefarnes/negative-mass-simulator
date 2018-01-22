#!/usr/bin/env python

""" vis.py: A script to enable visualisation of N-body simulations in image and video formats.
    PYTHONPATH=''
"""

import os
import numpy as np

import dask.array as da

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

from negmass_nbody.simulate.sim import find_existing, load_as_dask_array

__author__ = "Jamie Farnes"
__email__ = "jamie.farnes@oerc.ox.ac.uk"


def make_images():
    """A function to open and visualise an N-body simulation.
    
    Args:
    None
    
    Returns:
    None
    """
    # Load the parameters for the simulation:
    G, epsilon, chunks_value, limit, radius, time_steps = np.genfromtxt('./DATA/params.txt', dtype=float, usecols=[0, 1, 2, 3, 4, 5])
    sim_name = np.genfromtxt('./DATA/params.txt', dtype=str, usecols=[6])

    # Find the highest numbered HDF5 file:
    index = find_existing()
    print("Preparing to convert", index, ".hdf5 files into images...")

    img_steps = 0

    rotate = False

    # For structure formation with 50000 particles:
    if sim_name == "structure":
        alpha_input_neg = 0.09
        alpha_input_pos = 0.05

    # For Dark matter halo formation with 50000 particles:
    if sim_name == "halo":
        alpha_input_neg = 0.10
        alpha_input_pos = 0.017

    if rotate is True:
        # For structure formation with 50000 particles:
        if sim_name == "structure":
            eleim_initial = -90.0
            azim_initial = 0.0
        # For Dark matter halo formation with 50000 particles:
        if sim_name == "halo":
            eleim_initial = 30.0
            azim_initial = 30.0

    while img_steps < index:
        print("Processing iteration", img_steps, "...")
        # Load the data as dask arrays:
        position, velocity, mass = load_as_dask_array(img_steps, chunks_value)
        # Convert to numpy arrays:
        position = position.compute()
        velocity = velocity.compute()
        mass = mass.compute()
        # Make a 3D plot:
        plt.rcParams['axes.facecolor'] = 'black'
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_xlim(39920, 40080)
        ax.set_ylim(39920, 40080)
        ax.set_zlim(39920, 40080)
        # Plot the negative masses first:
        ax.scatter(position[mass < 0][:, 0], position[mass < 0][:, 1], position[mass < 0][:, 2], '.', s=1, color='purple', depthshade=True, alpha=alpha_input_neg)
        ax.scatter(position[mass < 0][:, 0], position[mass < 0][:, 1], position[mass < 0][:, 2], '.', s=0.5, color='purple', depthshade=True, alpha=alpha_input_neg-0.001)
        ax.scatter(position[mass < 0][:, 0], position[mass < 0][:, 1], position[mass < 0][:, 2], '.', s=0.1, color='purple', depthshade=True, alpha=alpha_input_neg-0.002)
        # Plot the positive masses now:
        ax.scatter(position[mass > 0][:, 0], position[mass > 0][:, 1], position[mass > 0][:, 2], '.', s=1, color='yellow', depthshade=True, alpha=alpha_input_pos)
        ax.scatter(position[mass > 0][:, 0], position[mass > 0][:, 1], position[mass > 0][:, 2], '.', s=0.5, color='lightyellow', depthshade=True, alpha=alpha_input_pos-0.001)
        ax.scatter(position[mass > 0][:, 0], position[mass > 0][:, 1], position[mass > 0][:, 2], '.', s=0.1, color='lightyellow', depthshade=True, alpha=alpha_input_pos-0.002)
        # Rotate the camera, if desired:
        if rotate is True:
            ax.view_init(elev=eleim_initial, azim=azim_initial)
            eleim_initial += 0.1
            azim_initial += 0.1
        # Save the plot to disk:
        plt.savefig('./DATA/img3d-' + str(img_steps) + '.png', bbox_inches='tight', dpi=300)
        plt.clf()
        plt.close()
        # Iterate:
        img_steps = img_steps+1

    return


def make_videos():
    """A function to make a video from the N-body .png images using ImageMagick.
    
    Args:
    None
    
    Returns:
    None
    """
    # Create directory to contain converted .jpgs, if needed:
    if os.path.isdir('./DATA/JPGS') is False:
        os.system('mkdir ./DATA/JPGS')

    # Find the highest numbered HDF5 file:
    index = find_existing()
    print("Preparing to convert images into an animated video...")

    # Loop over each file, adding the name to a list and converting to a .jpg:
    files = " "
    for i in range(0, index):
        if os.path.isfile("./DATA/img3d-" + str(i) + ".png") is True:
            files += "./DATA/JPGS/img3d-" + str(i) + ".jpg "
            os.system('/usr/local/bin/magick ' + './DATA/img3d-' + str(i) + '.png' + ' ./DATA/JPGS/img3d-' + str(i) + '.jpg')

    # Now combine all files together into an animation:
    os.system('/usr/local/bin/magick -delay 5 -loop 0' + files + './DATA/img3d-movie.mp4')
    print("Processing complete.")
    return
