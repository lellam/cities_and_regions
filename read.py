"""
Plot data in output folder.
"""


from urban_model import *
import numpy as np
import matplotlib.pyplot as plt

j = 0
# Read observation data
data = np.loadtxt("data/london_n/shopping_small.txt")
popn = np.loadtxt("data/london_n/popn.txt")
eP = popn[:, 2]/popn[:, 2].sum()
ret_locs = data[:, [0, 1]]
res_locs = popn[:, [0, 1]]
plt.figure(j)
plt.scatter(res_locs[:, 1], res_locs[:, 0], s=100*eP, alpha=0.5)
plt.scatter(ret_locs[:, 1], ret_locs[:, 0], color='r', s=1000*np.exp(xd))

# Low noise stats
samples = np.loadtxt("output/low_noise_samples.txt")
samples2 = np.loadtxt("output/low_noise_samples2.txt")
samples3 = np.loadtxt("output/low_noise_samples3.txt")
j+=1
plt.figure(j)
plt.title("Alpha low noise")
plt.plot(samples[:, 0])
plt.xlim([0, 20000])
j+=1
plt.figure(j)
plt.title("Beta low noise")
plt.plot(samples[:, 1])
plt.xlim([0, 20000])
print("\nLow noise stats:")
alpha_mean = np.dot(samples3, samples[:, 0])/np.sum(samples3)
alpha_sd = np.sqrt(np.dot(samples3, samples[:, 0]**2)/np.sum(samples3) - alpha_mean**2)
print("Alpha mean: " + str(alpha_mean))
print("Alpha sd: " + str(alpha_sd))
beta_mean = np.dot(samples3, samples[:, 1])/np.sum(samples3)
beta_sd = np.sqrt(np.dot(samples3, samples[:, 1]**2)/np.sum(samples3) - beta_mean**2)
print("Beta mean: " + str(beta_mean))
print("Beta sd: " + str(beta_sd))

x_e = (np.exp(samples2)*samples3[:, np.newaxis]).sum(axis=0)/np.sum(samples3)
x2_e = (np.exp(2*samples2)*samples3[:, np.newaxis]).sum(axis=0)/np.sum(samples3)
sd = np.sqrt(x2_e - x_e**2)
mean = x_e
lower = mean-3.*sd
upper = mean+3.*sd
j+=1
plt.figure(j)
plt.title("Low noise latents")
plt.scatter(res_locs[:, 1], res_locs[:, 0], s=100*eP, alpha=0.5)
plt.scatter(ret_locs[:, 1], ret_locs[:, 0], facecolors='none', edgecolors='r', s=1000*lower)
plt.scatter(ret_locs[:, 1], ret_locs[:, 0], facecolors='none', edgecolors='r', s=1000*upper)

# High noise stats
samples = np.loadtxt("output/high_noise_samples.txt")
samples2 = np.loadtxt("output/high_noise_samples2.txt")
samples3 = np.loadtxt("output/high_noise_samples3.txt")
j+=1
plt.figure(j)
plt.title("Alpha high noise")
plt.plot(samples[:, 0])
plt.xlim([0, 20000])
j+=1
plt.figure(j)
plt.title("Beta high noise")
plt.plot(samples[:, 1])
plt.xlim([0, 20000])
print("\nHigh noise stats:")
alpha_mean = np.dot(samples3, samples[:, 0])/np.sum(samples3)
alpha_sd = np.sqrt(np.dot(samples3, samples[:, 0]**2)/np.sum(samples3) - alpha_mean**2)
print("Alpha mean: " + str(alpha_mean))
print("Alpha sd: " + str(alpha_sd))
beta_mean = np.dot(samples3, samples[:, 1])/np.sum(samples3)
beta_sd = np.sqrt(np.dot(samples3, samples[:, 1]**2)/np.sum(samples3) - beta_mean**2)
print("Beta mean: " + str(beta_mean))
print("Beta sd: " + str(beta_sd))

x_e = (np.exp(samples2)*samples3[:, np.newaxis]).sum(axis=0)/np.sum(samples3)
x2_e = (np.exp(2*samples2)*samples3[:, np.newaxis]).sum(axis=0)/np.sum(samples3)
sd = np.sqrt(x2_e - x_e**2)
mean = x_e
lower = mean-3.*sd
upper = mean+3.*sd
j+=1
plt.figure(j)
plt.title("Low noise latents")
plt.scatter(res_locs[:, 1], res_locs[:, 0], s=100*eP, alpha=0.5)
plt.scatter(ret_locs[:, 1], ret_locs[:, 0], facecolors='none', edgecolors='r', s=1000*lower)
plt.scatter(ret_locs[:, 1], ret_locs[:, 0], facecolors='none', edgecolors='r', s=1000*upper)
plt.show()

# HMC plots
for alpha in [0.5, 1.0, 1.5, 2.0]:
    samples = np.loadtxt("output/hmc_samples" + str(alpha) + ".txt")
    xx = samples[-1]
    j += 1
    plt.figure(j)
    plt.title("HMC alpha=" + str(alpha))
    plt.scatter(res_locs[:, 1], res_locs[:, 0], s=100*eP, alpha=0.5)
    plt.scatter(ret_locs[:, 1], ret_locs[:, 0], color='r', s=1000*np.exp(xx))

# Opt plots
for alpha in [0.5, 1.0, 1.5, 2.0]:
    xx = np.loadtxt("output/opt" + str(alpha) + ".txt")
    j += 1
    plt.figure(j)
    plt.title("Opt alpha=" + str(alpha))
    plt.scatter(res_locs[:, 1], res_locs[:, 0], s=100*eP, alpha=0.5)
    plt.scatter(ret_locs[:, 1], ret_locs[:, 0], color='r', s=1000*np.exp(xx))

plt.show()