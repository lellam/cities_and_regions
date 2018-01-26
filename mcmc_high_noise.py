"""
MCMC scheme for high-noise regime.
"""

from urban_model import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import logsumexp
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

# Set theta for high-noise model
theta[3] = 100


# Load random stopping times
stopping = np.loadtxt("data/stopping.txt")


# Annealed importance sampling - returns an importance sampling estimate of z(theta)
def ais_ln_z(i):

    # Initialize AIS
    np.random.seed(None)
    p_n = 10                                # Number of samples
    t_n = 50                                # Number of bridging distributions
    L = 10                                  # HMC leapfrog steps
    eps = .1                                # HMC leapfrog stepsize
    temp = np.linspace(0, 1, t_n)   
    minustemp = 1. - temp
    ac = 0
    pc = 0
    log_weights = -np.log(p_n)*np.ones(p_n)

    delta = theta[2]
    gamma = theta[3]
    kappa = theta[4]

    # For each particle...
    for ip in range(p_n):
    
        # Initialize
        xx = np.log(np.random.gamma(gamma*(delta+1./mm), 1./(gamma*kappa), mm))     #Log-gamma model with alpha,beta->0
        V0, gradV0 = pot0_value(xx)
        V1, gradV1 = pot_value(xx)

        # Anneal...
        for it in range(1, t_n):
            log_weights[ip] += (temp[it] - temp[it-1])*(V0 - V1)

            # Initialize HMC
            p = np.random.normal(0., 1., mm)
            V, gradV = minustemp[it]*V0 + temp[it]*V1, minustemp[it]*gradV0 + temp[it]*gradV1
            H = 0.5*np.dot(p, p) + V

            # HMC leapfrog integrator
            x_p = xx
            p_p = p
            V_p, gradV_p = V, gradV
            for j in range(L):
                p_p = p_p - 0.5*eps*gradV_p
                x_p = x_p + eps*p_p
                V0_p, gradV0_p = pot0_value(x_p)
                V1_p, gradV1_p = pot_value(x_p)
                V_p, gradV_p = minustemp[it]*V0_p + temp[it]*V1_p, minustemp[it]*gradV0_p + temp[it]*gradV1_p
                p_p = p_p - 0.5*eps*gradV_p

            # HMC accept/reject
            pc+=1
            H_p = 0.5*np.dot(p_p, p_p) + V_p
            if np.log(np.random.uniform(0, 1)) < H - H_p:
                xx = x_p
                V0, gradV0 = V0_p, gradV0_p
                V1, gradV1 = V1_p, gradV1_p
                ac += 1

    return logsumexp(log_weights)


# Debiasing scheme - returns unbiased esimates of 1/z(theta)
def unbiased_z_inv(cc):

    N = int(stopping[cc])
    k_pow = 1.1
    print("Debiasing with N=" + str(N))

    # Get importance sampling estimate of z(theta) in parallel
    log_weights = Parallel(n_jobs=num_cores)(delayed(ais_ln_z)(i) for i in range(N+1))

    # Compute S = Y[0] + sum_i (Y[i] - Y[i-1])/P(K > i) using logarithms
    ln_Y = np.empty(N+1)
    ln_Y_pos = np.empty(N+1)
    ln_Y_neg = np.empty(N)

    for i in range(0, N+1):
        ln_Y[i] = np.log(i+1) - logsumexp(log_weights[:i+1])

    ln_Y_pos[0] = ln_Y[0]
    for i in range(1, N+1):
        ln_Y_pos[i] = ln_Y[i] + k_pow*np.log(i)
        ln_Y_neg[i-1] = ln_Y[i-1] + k_pow*np.log(i)

    positive_sum = logsumexp(ln_Y_pos)
    negative_sum = logsumexp(ln_Y_neg)

    ret = np.empty(2)
    if(positive_sum >= negative_sum):
        ret[0] = positive_sum + np.log(1. - np.exp(negative_sum - positive_sum))
        ret[1] = 1.
    else:
        ret[0] = negative_sum + np.log(1. - np.exp(positive_sum - negative_sum))
        ret[1] = -1.

    return ret


# MCMC tuning parameters
rwk_sd = 0.3                                                            #Randomwalk covariance
L2 = 50                                                                 #Number of leapfrog steps
eps2 = 0.02                                                             #Leapfrog step size


# Set-up MCMC
mcmc_start = 10000
mcmc_n = 20000

samples = np.empty((mcmc_n, 2))     # Theta-values
samples2 = np.empty((mcmc_n, mm))   # X-values
samples3 = np.empty(mcmc_n)         # Sign-values

samples_init = np.loadtxt("output/high_noise_samples.txt")
samples2_init = np.loadtxt("output/high_noise_samples2.txt")
samples3_init = np.loadtxt("output/high_noise_samples3.txt")

samples[:mcmc_start+1] = samples_init[:mcmc_start+1]
samples2[:mcmc_start+1] = samples2_init[:mcmc_start+1]
samples3[:mcmc_start+1] = samples3_init[:mcmc_start+1]


#Initialize MCMC
print("Starting at " + str(mcmc_start))
print("Warning max random stopping is " + str(stopping[mcmc_start:mcmc_n].max()))
kk = samples[mcmc_start]
xx = samples2[mcmc_start]
theta[0] = kk[0]
theta[1] = kk[1]*0.7e6
lnzinv, ss = unbiased_z_inv(mcmc_start-1)
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
    kk_p = kk + np.random.normal(0, rwk_sd, 2)
    for j in range(2):
        if kk_p[j] < 0.:
            kk_p[j] = -kk_p[j]
        elif kk_p[j] > 2.:
            kk_p[j] = 2. - (kk_p[j] - 2.)

    # Theta-accept/reject
    if kk_p.min() > 0 and kk_p.max() <= 2:
        theta[0] = kk_p[0]
        theta[1] = kk_p[1]*0.7e6
        lnzinv_p, ss_p = unbiased_z_inv(i)
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


    # Savedown and output details every 10 iterations
    if (i+1) % 10 == 0:
        print("Saving")
        np.savetxt("output/high_noise_samples.txt", samples)
        np.savetxt("output/high_noise_samples2.txt", samples2)
        np.savetxt("output/high_noise_samples3.txt", samples3)
        print("Theta AR " + str(float(ac)/float(pc)))
        print("X AR " + str(float(ac2)/float(pc2)))
        print("Net +ves " + str(np.sum(samples3[:(i+1)])))

print("Done")