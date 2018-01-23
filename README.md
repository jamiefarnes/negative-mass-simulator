Negative Mass N-body Simulation Codes
=====================================

This software allows one to perform three-dimensional (3D) gravitational N-body simulations in python, using both positive and negative masses.

These are corresponding codes for the paper "A Proposal for a Unifying Theory of Dark Energy and Dark Matter", available at:
http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1712.07962
and
https://arxiv.org/abs/1712.07962

Notes
-----

Most modern N-body software packages do not support exotic and rarely-studied phenomena such as negative masses. The code has therefore been deliberately written in order to achieve this goal.

The public release of these codes is intended to provide full transparency and to allow others to replicate and refine the findings in the aforementioned paper.

Negative masses are a creative explanation that require substantial scrutiny. The code therefore deliberately uses direct methods to evaluate the position and velocity of every particle at each timestep. This avoids the introduction of any approximations and maintains the highest accuracy. This thereby ensures that the measured effects are truly representative of a negative mass fluid and are not an artefact of any approximation. Nevertheless, this has a substantial cost in terms of computing time. 

Parallelisation has been included into the code, using Dask, in order to improve run times and to ensure that the code can run effectively on a standard laptop. Using a 2015 MacBook Pro, 3.1 GHz Intel Core i7, with 16 GB RAM, a single iteration with 50,000 particles takes ~3 minutes. A full run of ~1000 iterations requires ~50 hours of run time. If the code should need to be stopped for any reason, the N-body simulation code, run_nbody(), can simply be re-run, and it will automatically pick up from where it left off - beginning its calculation from the latest .hdf5 file saved to disk.

Dependencies
------------

numpy

matplotlib

h5py

Dask

ImageMagick
