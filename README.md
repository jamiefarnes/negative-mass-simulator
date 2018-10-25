Negative Mass N-body Simulation Codes
=====================================

This software allows one to perform three-dimensional (3D) gravitational N-body simulations in python, using both positive and negative masses.

These are corresponding codes for the paper "A unifying theory of dark energy and dark matter: Negative masses and matter creation within a modified LambdaCDM framework", available at:
http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1712.07962
https://arxiv.org/abs/1712.07962

The paper has been accepted for publication in Astronomy and Astrophysics (A&A).

Notes
-----

Most modern N-body software packages do not support exotic and rarely-studied phenomena such as negative masses. The code has therefore been deliberately written in order to achieve this goal. I have therefore written new software to perform three-dimensional (3D) gravitational N-body simulations using python, numpy, and matplotlib. The code is parallelised using Dask in order to make use of the multiple processing cores available in most modern machines. 

The primary motivation for this computational perspective is not focussed on performance, but rather on providing easily-understandable, open-source software. This enables the presented results to be easily replicated and verified on any scientist's own machine, rather than requiring any specialised hardware or software setup. The simulations presented here are therefore necessarily primitive in comparison with the state-of-the-art, but provide examples that demonstrate what can be expected from such a toy model. 

Negative masses are a creative explanation that require substantial scrutiny. The code therefore deliberately uses direct methods to evaluate the position and velocity of every particle at each timestep. This avoids the introduction of any approximations and maintains the highest accuracy. Nevertheless, this has a substantial cost in terms of computing time. 

Using a 2015 MacBook Pro, 3.1 GHz Intel Core i7, with 16 GB RAM, a single iteration with 50,000 particles takes ~3 minutes. A full run of ~1000 iterations requires ~50 hours of run time. If the code should need to be stopped for any reason, the N-body simulation code, run_nbody(), can simply be re-run, and it will automatically pick up from where it left off - beginning its calculation from the latest .hdf5 file saved to disk.

Dependencies
------------

numpy

matplotlib

h5py

Dask

ImageMagick
