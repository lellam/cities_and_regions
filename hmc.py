"""
HMC scheme to sample from prior for latent variables.
"""

from urban_model import *
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Set theta for high-noise model
theta[0] = 2.0
theta[1] = 0.3*0.7e6
theta[2] = 0.3/mm
theta[3] = 100.
theta[4] = 1.3

# MCMC tuning parameters
L = 10                                                                 #Number of leapfrog steps
eps = 0.1                                                              #Leapfrog step size


# Set-up MCMC
mcmc_n = 10000
temp_n = 5
inverse_temps = np.array([1., 1./2., 1./4., 1./8., 1./16.])
samples = np.empty((mcmc_n, mm))   # X-values


#Initialize MCMC
xx = -np.log(mm)*np.ones((temp_n, mm))
V = np.empty(temp_n)
gradV = np.empty((temp_n, mm))
for j in range(temp_n):
    V[j], gradV[j] = pot_value(xx[j])


# Counts to keep track of accept rates
ac = np.zeros(temp_n)
pc = np.zeros(temp_n)
acs = 0
pcs = 1

# MCMC algorithm
for i in range(mcmc_n):

    for j in range(temp_n):
        #Initialize leapfrog integrator for HMC proposal
        p = np.random.normal(0., 1., mm)

        H = 0.5*np.dot(p, p) + inverse_temps[j]*V[j]


        # X-Proposal
        x_p = xx[j]
        p_p = p
        V_p, gradV_p = V[j], gradV[j]
        for l in range(L):
            p_p = p_p -0.5*eps*inverse_temps[j]*gradV_p
            x_p = x_p + eps*p_p
            V_p, gradV_p = pot_value(x_p)
            p_p = p_p - 0.5*eps*inverse_temps[j]*gradV_p

        # X-accept/reject
        pc[j] += 1
        H_p = 0.5*np.dot(p_p, p_p) + inverse_temps[j]*V_p
        if np.log(np.random.uniform(0, 1)) < H - H_p:
            xx[j] = x_p
            V[j], gradV[j] = V_p, gradV_p
            ac[j] += 1

    # Perform a swap
    pcs += 1
    j0 = np.random.randint(0, temp_n-1)
    j1 = j0+1
    logA = (inverse_temps[j1]-inverse_temps[j0])*(-V[j1] + V[j0])
    if np.log(np.random.uniform(0, 1)) < logA:
        xx[[j0, j1]] = xx[[j1, j0]]
        V[[j0, j1]] = V[[j1, j0]]
        gradV[[j0, j1]] = gradV[[j1, j0]]
        acs += 1

    # Update stored Markov-chain
    samples[i] = xx[0]

    # Savedown and output details every 100 iterations
    if (i+1) % 100 == 0:
        print("Saving iteration " + str(i+1))
        np.savetxt("output/hmc_samples" + str(theta[0]) + ".txt", samples)
        print("X AR:")
        print(ac/pc)
        print("Swap AR:" + str(float(acs)/float(pcs)))

print("Done")