#!/usr/bin/env python

"""init.py: Functions for initialising various simulations."""

import os
import numpy as np

from negmass_nbody.export.save import save_data

__author__ = "Jamie Farnes"
__email__ = "jamie.farnes@oerc.ox.ac.uk"


def initialise_halo_params():
    """Initialise the basic parameters needed to simulate a forming Dark matter halo.
        
    Args:
    None
    
    Returns:
    G: gravitational constant.
    epsilon: softening parameter.
    limit: width of the simulated universe.
    radius: simulated radius of each particle
    (for proper handling of boundary conditions).
    num_pos_particles: number of positive mass particles.
    num_neg_particles: number of negative mass particles.
    chunks_value: dask chunks value.
    time_steps: number of time steps to simulate.
    """
    G = 1.0
    epsilon = 0.07
    limit = 80000
    radius = 4
    num_pos_particles = 5000
    num_neg_particles = 45000
    chunks_value = (num_pos_particles+num_neg_particles)/5.0
    time_steps = 1000
    return G, epsilon, limit, radius, num_pos_particles, num_neg_particles, chunks_value, time_steps


def initialise_halo_sim():
    """Initialise the specific parameters needed to simulate a forming Dark matter halo.
    
    Args:
    None
    
    Returns:
    M_pos: the total positive mass in the simulated universe.
    M_neg: the total negative mass in the simulated universe.
    a_scale: the Hernquist scale radius of the positive mass galaxy.
    gauss_vel_comp: the small Gaussian velocity component of the positive mass galaxy.
    Set to zero to build cold.
    cube_neg_width: width of the uniformly distributed negative mass cube.
    sim_name: name of the simulation.
    """
    M_pos = 1.0
    M_neg = -3.0
    a_scale = 1.0
    gauss_vel_comp = 0.3
    cube_neg_width = 200
    sim_name = "halo"
    return M_pos, M_neg, a_scale, gauss_vel_comp, cube_neg_width, sim_name


def hernquist_ppf(r, a_scale=1.0):
    """Return a basic Hernquist ppf.
    
    Args:
    r (numpy array): array of random numbers.
    a (float): Hernquist scale radius.
    
    Returns:
    ppf: basic Hernquist ppf.
    """
    ppf = (a_scale-(a_scale*r)+np.sqrt(a_scale**2 - (r*(a_scale**2))))/r
    return ppf


def hernquist_vcirc(r, a_scale=1.0, m=1.0, G=1.0):
    """Return a basic Hernquist circular orbital velocity.
    
    Args:
    r (numpy array): array of radii in spherical coordinates.
    a (float): Hernquist scale radius.
    m (numpy array): array of masses for all positive mass particles.
    G (float): gravitational constant.
    
    Returns:
    v_circ: basic Hernquist circular orbital velocity.
    """
    v_circ = (np.sqrt(G*m*r*((r+a_scale)**(-2))))
    return v_circ


def particle_halo_init(G, num_pos_particles, num_neg_particles, num_tot_particles, limit, m_pos, m_neg, a_scale, gauss_vel_comp, cube_neg_width):
    """Initialise the positions, velocities, and masses of all particles for a forming Dark matter halo simulation.
    
    Args:
    G (float): gravitational constant.
    num_pos_particles (float): number of positive mass particles.
    num_neg_particles (float): number of negative mass particles.
    num_tot_particles (float): total number of particles.
    limit (float): width of the simulated universe.
    m_pos (float): the total positive mass in the simulated universe.
    m_neg (float): the total negative mass in the simulated universe.
    a_scale (float): the Hernquist scale radius of the positive mass galaxy.
    gauss_vel_comp (float): the small Gaussian velocity component of the positive mass galaxy.
    cube_neg_width (float): width of the uniformly distributed negative mass cube.
    
    Returns:
    position: numpy array of all particle positions in cartesian coordinates.
    velocity: numpy array of all particle velocities in cartesian coordinates.
    mass: numpy array of all particle masses.
    """
    # Set all the masses:
    if num_pos_particles > 0:
        mass_pos = np.random.uniform(m_pos/num_pos_particles, m_pos/num_pos_particles, num_pos_particles)
    else:
        mass_pos = np.array([])
    if num_neg_particles > 0:
        mass_neg = np.random.uniform(m_neg/num_neg_particles, m_neg/num_neg_particles, num_neg_particles)
    else:
        mass_neg = np.array([])
    mass = np.concatenate((mass_pos, mass_neg), axis=0)
    if len(mass) == 0:
        print("ERROR: No particles included in the simulation.")
    # Initially set all velocities to zero:
    velocity = 0.0*np.random.randn(num_tot_particles, 3)
    # For the positive masses (distributed as a central Hernquist galaxy):
    # Generate an array of random positions in spherical coordinates:
    r = hernquist_ppf(np.random.uniform(0, 1, num_pos_particles), a_scale)
    phi = np.random.uniform(0, 2*np.pi, num_pos_particles)
    theta = np.arccos(np.random.uniform(-1, 1, num_pos_particles))
    # Convert to cartesian coordinates (located at the centre of the simulation):
    x = r*np.sin(theta)*np.cos(phi) + limit/2.0
    y = r*np.sin(theta)*np.sin(phi) + limit/2.0
    z = r*np.cos(theta) + limit/2.0
    # Generate phi and theta a second time (otherwise all velocities will be
    # radial, with no tangential component):
    phi_v = np.random.uniform(0, 2*np.pi, num_pos_particles)
    theta_v = np.arccos(np.random.uniform(-1, 1, num_pos_particles))
    for i in range(num_pos_particles):
        vel_0 = hernquist_vcirc(r, a_scale, mass[0:num_pos_particles], G)
        velocity[i][0] = vel_0[i]*np.sin(theta_v[i])*np.cos(phi_v[i])+np.random.normal(0.0, gauss_vel_comp, 1)
        velocity[i][1] = vel_0[i]*np.sin(theta_v[i])*np.sin(phi_v[i])+np.random.normal(0.0, gauss_vel_comp, 1)
        velocity[i][2] = vel_0[i]*np.cos(theta_v[i])+np.random.normal(0.0, gauss_vel_comp, 1)
    # For the negative masses (distributed as a uniformly distributed cube):
    x_neg = np.random.uniform((limit/2.0)-(cube_neg_width/2.0), (limit/2.0)+(cube_neg_width/2.0), num_neg_particles)
    y_neg = np.random.uniform((limit/2.0)-(cube_neg_width/2.0), (limit/2.0)+(cube_neg_width/2.0), num_neg_particles)
    z_neg = np.random.uniform((limit/2.0)-(cube_neg_width/2.0), (limit/2.0)+(cube_neg_width/2.0), num_neg_particles)
    # Combine the positive and negative masses together:
    x = np.concatenate((x, x_neg), axis=0)
    y = np.concatenate((y, y_neg), axis=0)
    z = np.concatenate((z, z_neg), axis=0)
    position = np.column_stack((x, y, z))
    # Set the type to float32, in order to reduce the memory requirements:
    position = position.astype(np.float32, copy=False)
    velocity = velocity.astype(np.float32, copy=False)
    mass = mass.astype(np.float32, copy=False)
    return position, velocity, mass


def initialise_structure_params():
    """Initialise the basic parameters needed to simulate structure formation.
    
    Args:
    None
    
    Returns:
    G: gravitational constant.
    epsilon: softening parameter.
    limit: width of the simulated universe.
    radius: simulated radius of each particle
    (for proper handling of boundary conditions).
    num_pos_particles: number of positive mass particles.
    num_neg_particles: number of negative mass particles.
    chunks_value: dask chunks value.
    time_steps: number of time steps to simulate.
    """
    G = 1.0
    epsilon = 0.07  # softening parameter
    limit = 80000
    radius = 4
    num_pos_particles = 25000
    num_neg_particles = 25000
    chunks_value = (num_pos_particles+num_neg_particles)/5.0
    time_steps = 1000
    return G, epsilon, limit, radius, num_pos_particles, num_neg_particles, chunks_value, time_steps


def initialise_structure_sim():
    """Initialise the specific parameters needed to simulate structure formation.
    
    Args:
    None
    
    Returns:
    M_pos: the total positive mass in the simulated universe.
    M_neg: the total negative mass in the simulated universe.
    cube_pos_width: width of the uniformly distributed positive mass cube.
    cube_neg_width: width of the uniformly distributed negative mass cube.
    sim_name: name of the simulation.
    """
    M_pos = 1.0
    M_neg = -1.0
    cube_pos_width = 200  # positive mass setting, width of uniformly distributed cube
    cube_neg_width = 200  # negative mass setting, width of uniformly distributed cube
    sim_name = "structure"
    return M_pos, M_neg, cube_pos_width, cube_neg_width, sim_name


def particle_structure_init(G, num_pos_particles, num_neg_particles, num_tot_particles, limit, m_pos, m_neg, cube_pos_width, cube_neg_width):
    """Initialise the positions, velocities, and masses of all particles for a structure formation simulation.
    
    Args:
    G (float): gravitational constant.
    num_pos_particles (float): number of positive mass particles.
    num_neg_particles (float): number of negative mass particles.
    num_tot_particles (float): total number of particles.
    limit (float): width of the simulated universe.
    m_pos (float): the total positive mass in the simulated universe.
    m_neg (float): the total negative mass in the simulated universe.
    cube_pos_width (float): width of the uniformly distributed positive mass cube.
    cube_neg_width (float): width of the uniformly distributed negative mass cube.
    
    Returns:
    position: numpy array of all particle positions in cartesian coordinates.
    velocity: numpy array of all particle velocities in cartesian coordinates.
    mass: numpy array of all particle masses.
    """
    # Set all the masses:
    if num_pos_particles > 0:
        mass_pos = np.random.uniform(m_pos/num_pos_particles, m_pos/num_pos_particles, num_pos_particles)
    else:
        mass_pos = np.array([])
    if num_neg_particles > 0:
        mass_neg = np.random.uniform(m_neg/num_neg_particles, m_neg/num_neg_particles, num_neg_particles)
    else:
        mass_neg = np.array([])
    mass = np.concatenate((mass_pos, mass_neg), axis=0)
    if len(mass) == 0:
        print("ERROR: No particles included in the simulation.")
    # Initially set all velocities to zero:
    velocity = 0.0*np.random.randn(num_tot_particles, 3)
    # For the positive masses (distributed as a uniformly distributed cube):
    x = np.random.uniform((limit/2.0)-(cube_pos_width/2.0), (limit/2.0)+(cube_pos_width/2.0), num_pos_particles)
    y = np.random.uniform((limit/2.0)-(cube_pos_width/2.0), (limit/2.0)+(cube_pos_width/2.0), num_pos_particles)
    z = np.random.uniform((limit/2.0)-(cube_pos_width/2.0), (limit/2.0)+(cube_pos_width/2.0), num_pos_particles)
    # For the negative masses (distributed as a uniformly distributed cube):
    x_neg = np.random.uniform((limit/2.0)-(cube_neg_width/2.0), (limit/2.0)+(cube_neg_width/2.0), num_neg_particles)
    y_neg = np.random.uniform((limit/2.0)-(cube_neg_width/2.0), (limit/2.0)+(cube_neg_width/2.0), num_neg_particles)
    z_neg = np.random.uniform((limit/2.0)-(cube_neg_width/2.0), (limit/2.0)+(cube_neg_width/2.0), num_neg_particles)
    # Combine the positive and negative masses together:
    x = np.concatenate((x, x_neg), axis=0)
    y = np.concatenate((y, y_neg), axis=0)
    z = np.concatenate((z, z_neg), axis=0)
    position = np.column_stack((x, y, z))
    # Set the type to float32, in order to reduce the memory requirements:
    position = position.astype(np.float32, copy=False)
    velocity = velocity.astype(np.float32, copy=False)
    mass = mass.astype(np.float32, copy=False)
    return position, velocity, mass


def init_dm_halo():
    """A function to initiate and setup an N-body simulation of a forming Dark matter halo.
    
    Args:
    None
    
    Returns:
    None
    """
    print("Initialising Dark matter halo formation simulation...")
    
    # Clean up any files from previous runs:
    if os.path.isdir('DATA') is False:
        os.system('mkdir DATA')
    os.system('rm -rf ./DATA/*.hdf5')

    # Initialise the basic parameters to use:
    G, epsilon, limit, radius, num_pos_particles, num_neg_particles, chunks_value, time_steps = initialise_halo_params()
    num_tot_particles = num_pos_particles+num_neg_particles
    
    # Initialise the specific parameters to use for the simulation:
    m_pos, m_neg, a_scale, gauss_vel_comp, cube_neg_width, sim_name = initialise_halo_sim()
    
    # Initialise particle positions, velocities, and masses:
    position, velocity, mass = particle_halo_init(G, num_pos_particles, num_neg_particles, num_tot_particles, limit, m_pos, m_neg, a_scale, gauss_vel_comp, cube_neg_width)
    
    # Save the data to hdf5 format and the parameters to a .txt file:
    print("Saving initial conditions to disk...")
    save_data(position, './DATA/position0.hdf5', chunks_value)
    save_data(velocity, './DATA/velocity0.hdf5', chunks_value)
    save_data(mass, './DATA/mass.hdf5', chunks_value)
    
    np.savetxt('./DATA/params.txt', np.column_stack([G, epsilon, chunks_value, limit, radius, time_steps, sim_name]), fmt="%s")

    return


def init_structure_formation():
    """A function to initiate and setup an N-body simulation of structure formation.
    
    Args:
    None
    
    Returns:
    None
    """
    print("Initialising structure formation simulation...")
    
    # Clean up any files from previous runs:
    if os.path.isdir('DATA') is False:
        os.system('mkdir DATA')
    os.system('rm -rf ./DATA/*.hdf5')

    # Initialise the basic parameters to use:
    G, epsilon, limit, radius, num_pos_particles, num_neg_particles, chunks_value, time_steps = initialise_structure_params()
    num_tot_particles = num_pos_particles+num_neg_particles
    
    # Initialise the specific parameters to use for the simulation:
    m_pos, m_neg, cube_pos_width, cube_neg_width, sim_name = initialise_structure_sim()
    
    # Initialise particle positions, velocities, and masses:
    position, velocity, mass = particle_structure_init(G, num_pos_particles, num_neg_particles, num_tot_particles, limit, m_pos, m_neg, cube_pos_width, cube_neg_width)
    
    # Save the data to hdf5 format and the parameters to a .txt file:
    print("Saving initial conditions to disk...")
    save_data(position, './DATA/position0.hdf5', chunks_value)
    save_data(velocity, './DATA/velocity0.hdf5', chunks_value)
    save_data(mass, './DATA/mass.hdf5', chunks_value)
    
    np.savetxt('./DATA/params.txt', np.column_stack([G, epsilon, chunks_value, limit, radius, time_steps, sim_name]), fmt="%s")
    
    return
