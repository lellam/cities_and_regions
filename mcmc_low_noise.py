"""
MCMC scheme for low-noise regime.
"""

from urban_model import *
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Set theta for low-noise model
theta[3] = 10000


# To compute 1/z(theta) using Laplace approximation
def z_inv():
    m = xd
    f_val = np.infty
    for k in range(mm):
        delta = theta[2]
        g = np.log(delta)*np.ones(mm)
        g[k] = np.log(1.+delta)
        f = minimize(pot_value,g, method='L-BFGS-B', jac=True, options={'disp': False})
        if(f.fun < f_val):
            f_val = f.fun
            m = f.x
    A = pot_hess(m)
    L = np.linalg.cholesky(A)
    half_log_det_A = np.sum(np.log(np.diag(L)))
    ret = np.empty(2)
    ret[0] = pot_value(m)[0] +  half_log_det_A
    ret[1] = 1.
    return ret


# MCMC tuning parameters
Ap = np.array([[ 0.00749674,  0.00182529], [ 0.00182529,  0.00709968]]) #Randomwalk covariance
L2 = 50                                                                 #Number of leapfrog steps
eps2 = 0.02                                                             #Leapfrog step size


# Set-up MCMC
mcmc_start = 1
mcmc_n = 20000

samples = np.empty((mcmc_n, 2))     # Theta-values
samples2 = np.empty((mcmc_n, mm))   # X-values
samples3 = np.empty(mcmc_n)         # Sign-values

samples_init = np.loadtxt("output/low_noise_samples.txt")
samples2_init = np.loadtxt("output/low_noise_samples2.txt")
samples3_init = np.loadtxt("output/low_noise_samples3.txt")

samples[:mcmc_start+1] = samples_init[:mcmc_start+1]
samples2[:mcmc_start+1] = samples2_init[:mcmc_start+1]
samples3[:mcmc_start+1] = samples3_init[:mcmc_start+1]


#Initialize MCMC
print("Starting at " + str(mcmc_start))
kk = samples[mcmc_start]
xx = samples2[mcmc_start]
theta[0] = kk[0]
theta[1] = kk[1]*0.7e6
lnzinv, ss = z_inv()
V, gradV = pot_value(xx)


# Counts to keep track of accept rates
ac = 0
pc = 0
ac2 = 0
pc2 = 0


# MCMC algorithm
for i in range(mcmc_start, mcmc_n):

    print("\nIteration:" + str(i))

    # Theta-proposal (random walk with reflecting boundaries)
    kk_p = kk + 1.*np.dot(Ap, np.random.normal(0, 1, 2))
    for j in range(2):
        if kk_p[j] < 0.:
            kk_p[j] = -kk_p[j]
        elif kk_p[j] > 2.:
            kk_p[j] = 2. - (kk_p[j] - 2.)

    # Theta-accept/reject
    if kk_p.min() > 0 and kk_p.max() <= 2:
        try:
            theta[0] = kk_p[0]
            theta[1] = kk_p[1]*0.7e6
            lnzinv_p, ss_p = z_inv()
            V_p, gradV_p = pot_value(xx)
            pp_p = lnzinv_p - V_p
            pp = lnzinv - V
            print("Proposing " + str(kk_p) + " with " + str(ss_p))
            print(str(pp_p) + " vs " + str(pp))

            pc += 1
            if np.log(np.random.uniform(0, 1)) < pp_p - pp:
                print("Theta-Accept")
                kk = kk_p
                V, gradV = V_p, gradV_p
                ac += 1
                lnzinv, ss = lnzinv_p, ss_p
                
            else:
                print("Theta-Reject")
        except:
            None


    # Reset theta for HMC
    theta[0] = kk[0]
    theta[1] = kk[1]*0.7e6


    #Initialize leapfrog integrator for HMC proposal
    p = np.random.normal(0., 1., mm)
    VL, gradVL = like_value(xx)
    W, gradW = V + VL, gradV + gradVL
    H = 0.5*np.dot(p, p) + W


    # X-Proposal
    x_p = xx
    p_p = p
    W_p, gradW_p = W, gradW
    for j in range(L2):
        p_p = p_p -0.5*eps2*gradW_p
        x_p = x_p + eps2*p_p
        VL_p, gradVL_p = like_value(x_p)
        V_p, gradV_p = pot_value(x_p)
        W_p, gradW_p = V_p + VL_p, gradV_p + gradVL_p
        p_p = p_p - 0.5*eps2*gradW_p

    # X-accept/reject
    pc2 += 1
    H_p = 0.5*np.dot(p_p, p_p) + W_p
    if np.log(np.random.uniform(0, 1)) < H - H_p:
        xx = x_p
        V, gradV = V_p, gradV_p
        ac2 += 1
        print("X-Accept")
    else:
        print("X-Reject")

    # Update stored Markov-chain
    samples[i] = kk
    samples2[i] = xx
    samples3[i] = ss

    # Savedown and output details every 100 iterations
    if (i+1) % 100 == 0:
        print("Saving")
        np.savetxt("output/low_noise_samples.txt", samples)
        np.savetxt("output/low_noise_samples2.txt", samples2)
        np.savetxt("output/low_noise_samples3.txt", samples3)
        print("Theta AR " + str(float(ac)/float(pc)))
        print("X AR " + str(float(ac2)/float(pc2)))
        print("Net +ves " + str(np.sum(samples3[:(i+1)])))

print("Done")