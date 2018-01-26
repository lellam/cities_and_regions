Data and code for "Stochastic Modelling of Urban Structure" manuscript

To run any of the Python files, potential_functions.c must be compiled into a shared object called potential_functions.so in the same folder.  e.g. on OSX this can be done from the terminal with:
gcc -fPIC -shared -o potential.so potential.c -O3

Examples were run with Python 3.6.0. with recent versions of numpy, scipy, matplotlib, ctypes, joblib and multiprocessing libraries installed.

The following files relate to figures in the manuscript:

- hmc.py
HMC code to generate data for figure 5.

- laplace_grid.py
Likelihood values for figure 4.

- mcmc_high_noise.py:
MCMC scheme for gamma=10000 (low-noise regime) to generate data for figures 9-10.

- mcmc_low_noise.py:
MCMC scheme for gamma=100 (high-noise regime) to generate data for figures 7-8.

- opt.py:
Optimization routine to generate data for figure 6.

- potential_2d.py:
Illustration of 2d potential function to produce figure 2.

- read.py:
Plot MCMC and optimisation data saved down into the output directory.  Will produce similar to figures 3, 5, 6, 8, 10 and stats relating to figures 7 and 9.

- rsquared_analysis.py:
R-squared analysis for deterministic model as discussed around figure 4.

- data/london_n/
Directory containing datasets for the case study.  Residential data is residential.csv and retail data is small_london.txt.  Remaining files are pre-processed versions for simulations.