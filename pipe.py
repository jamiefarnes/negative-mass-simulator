#!/usr/bin/env python

""" pipe.py: A script to demonstrate a standard run of an N-body simulation.
    PYTHONPATH=''
"""

from negmass_nbody.initialise.init import init_dm_halo, init_structure_formation
from negmass_nbody.simulate.sim import run_nbody
from negmass_nbody.visualise.vis import make_images, make_videos

__author__ = "Jamie Farnes"
__email__ = "jamie.farnes@oerc.ox.ac.uk"


# Setup the simulation for either halo or structure formation:
SPECIFY_SIM = 'halo'

if SPECIFY_SIM == 'halo':
    init_dm_halo()
elif SPECIFY_SIM == 'structure':
    init_structure_formation()
else:
    print("ERROR: Unknown simulation type.")

# Create the snapshots:
run_nbody()

# Make images from the data:
make_images()

# Concatenate images into video format:
make_videos()
