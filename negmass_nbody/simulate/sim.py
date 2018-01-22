#!/usr/bin/env python

"""sim.py: Functions for running the N-body code."""

import os
import time as t
import numpy as np

import h5py
import dask.array as da

from negmass_nbody.export.save import save_data

__author__ = "Jamie Farnes"
__email__ = "jamie.farnes@oerc.ox.ac.uk"


def find_existing():
    """Find the highest iteration of the simulation that has been processed thus far.
    
    Args:
    None
    
    Returns:
    index: the number of the highest iteration.
    """
    listing = os.listdir('./DATA/')
    filtered_list = [listing[i].split('.hdf5')[0].split('velocity') for i in range(len(listing))]
    index = []
    for i in range(len(filtered_list)):
        if len(filtered_list[i]) == 2:
            if len(filtered_list[i][1]) > 0:
                index.append(int(filtered_list[i][1]))
    index = np.max(np.array(index))
    return index


def load_as_dask_array(index, chunks_value):
    """Load a specific iteration of the simulation as a Dask array.
    
    Args:
    index (float): the number of the highest iteration.
    chunks_value (float): dask chunks value.
    
    Returns:
    position: dask array of all particle positions in cartesian coordinates.
    velocity: dask array of all particle velocities in cartesian coordinates.
    mass: dask array of all particle masses.
    """
    file1 = h5py.File('./DATA/position' + str(index) + '.hdf5', mode='r')
    position_dset = file1['/x']  # refer to the array on disk
    file2 = h5py.File('./DATA/velocity' + str(index) + '.hdf5', mode='r')
    velocity_dset = file2['/x']
    file3 = h5py.File('./DATA/mass.hdf5', mode='r')
    mass_dset = file3['/x']
    # Create dask arrays:
    position = da.from_array(position_dset, chunks=(chunks_value))
    velocity = da.from_array(velocity_dset, chunks=(chunks_value))
    mass = da.from_array(mass_dset, chunks=(chunks_value))
    return position, velocity, mass


def update_velocities(position, velocity, mass, G, epsilon):
    """Calculate the interactions between all particles and update the velocities.
    
    Args:
    position (dask array): dask array of all particle positions in cartesian coordinates.
    velocity (dask array): dask array of all particle velocities in cartesian coordinates.
    mass (dask array): dask array of all particle masses.
    G (float): gravitational constant.
    epsilon (float): softening parameter.
    
    Returns:
    velocity: updated particle velocities in cartesian coordinates.
    """
    dx = da.subtract.outer(position[:, 0], position[:, 0])
    dy = da.subtract.outer(position[:, 1], position[:, 1])
    dz = da.subtract.outer(position[:, 2], position[:, 2])
    r2 = da.square(dx) + da.square(dy) + da.square(dz) + da.square(epsilon)
    #
    coef = -G*mass[:]
    ax = coef*dx
    ay = coef*dy
    az = coef*dz
    #
    ax_scaled = da.divide(ax, r2)
    ay_scaled = da.divide(ay, r2)
    az_scaled = da.divide(az, r2)
    #
    total_ax = da.nansum(ax_scaled, axis=1)
    total_ay = da.nansum(ay_scaled, axis=1)
    total_az = da.nansum(az_scaled, axis=1)
    #
    velocity_x = da.diag(da.add.outer(da.transpose(velocity)[0], total_ax))
    velocity_y = da.diag(da.add.outer(da.transpose(velocity)[1], total_ay))
    velocity_z = da.diag(da.add.outer(da.transpose(velocity)[2], total_az))
    #
    velocity = np.column_stack((velocity_x.compute(), velocity_y.compute(), velocity_z.compute()))
    return velocity


def apply_boundary_conditions(position, velocity, limit, radius):
    """Calculate the interactions between all particles and update the velocities.
    
    Args:
    position (numpy array): numpy array of all particle positions in cartesian coordinates.
    velocity (numpy array): numpy array of all particle velocities in cartesian coordinates.
    limit (float): width of the simulated universe.
    radius (float): simulated radius of each particle
    (for proper handling of boundary conditions).
    
    Returns:
    position: updated particle positions in cartesian coordinates.
    velocity: updated particle velocities in cartesian coordinates.
    """
    # Masks:
    xmax = position[:, 0] > limit
    xmin = position[:, 0] < 0
    ymax = position[:, 1] > limit
    ymin = position[:, 1] < 0
    zmax = position[:, 2] > limit
    zmin = position[:, 2] < 0
    # Flip velocities at boundary (only a concern if particles reach the boundary):
    velocity[xmax | xmin, 0] *= -1.0
    velocity[ymax | ymin, 1] *= -1.0
    velocity[zmax | zmin, 2] *= -1.0
    # Clip motion to bounding box:
    position[xmax, 0] = limit - 2 * radius
    position[xmin, 0] = 2 * radius
    position[ymax, 1] = limit - 2 * radius
    position[ymin, 1] = 2 * radius
    position[zmax, 2] = limit - 2 * radius
    position[zmin, 2] = 2 * radius
    return position, velocity


def run_nbody():
    """A function to run and continue iterating an N-body simulation.
        
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
    print("Beginning N-body simulation. Iteration:", index, "of", int(time_steps))
    
    time_steps = time_steps-index
    
    while time_steps > 0:
        # Load the data as dask arrays:
        position, velocity, mass = load_as_dask_array(index, chunks_value)
        
        # Start the N-body iteration:
        start = t.time()
        
        # Update the particle velocities:
        velocity = update_velocities(position, velocity, mass, G, epsilon)
        
        # Update the particle positions:
        position = position.compute()
        position += velocity
        
        # End the iteration:
        end = t.time()
        print("Iteration", index, "complete:", (end-start)/60.0, "mins.")
        
        # Apply various boundary conditions:
        position, velocity = apply_boundary_conditions(position, velocity, limit, radius)
        
        # Save the data:
        save_data(position, './data/position' + str(index+1) + '.hdf5', chunks_value)
        save_data(velocity, './data/velocity' + str(index+1) + '.hdf5', chunks_value)
        index = index+1
        time_steps = time_steps-1
    
    return
