# python-noiselab

noiselab is a collection of Cython- or C-optimized implementations of algorithms for the simulation and analysis of
continuous-variable random processes in Python. In particular, this library is currently focused on the simulation of
stochastic signals in time with non-trivial temporal correlations and distributions.

There are a collection of Generator classes, which can generate continuous streams of stochastic signals with various
power spectra. These noisy signals can then be analyzed using Power Spectral Density estimation routines in noiselab.psd
or the Allan deviation, in noiselab.adev. This library also contains a wrapper for a fast TDigest library written in C,
which can be used to efficiently estimate quantiles of arbitrary noise distributions.

# Generators

* WhitenoiseProcess: White noise
* WienerProcess: Brownian noise / Wiener process
* MarkovProcess: Orenstein-Uhlenbeck / 1st-order Markov process
* RelaxationProcess: $\alpha = 0 - 2$ Power-law noise (PLNoise)

## RelaxationProcess - PLNoise

Fast Cython implemention of PLNoise2 power-low noise generation algorithm described in the following references:

Milotti, Edoardo (2005), "Exact numerical simulation of power-law noises", Physical Review E, Volume 72, Issue 5, https://doi.org/10.1103/PhysRevE.72.056701 (https://arxiv.org/abs/physics/0509237)

Milotti, Edoardo (2007), "New version of PLNoise: a package for exact numerical simulation of power-law noises", Computer Physics Communications, Volume 177, Issue 4
Link: https://doi.org/10.1016/j.cpc.2007.04.005

Milotti, Edoardo (2007), “New version of PLNoise: a package for exact numerical simulation of power-law noises ”, Mendeley Data, V1, doi: 10.17632/nt7mcwh98m.1
Link: https://doi.org/10.17632/nt7mcwh98m.1


# TDigest
* https://arxiv.org/abs/1902.04023
* Python bindings for C library: https://github.com/RedisBloom/t-digest-c
