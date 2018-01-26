"""
2D illustration of potential function
"""

from urban_model import *
import numpy as np
import matplotlib.pyplot as plt


# Setup 2D model
cost_mat = cost_mat[:,:2]/cost_mat[:,:2].sum()
nn, mm = np.shape(cost_mat)

alpha = 0.5
beta = 1000
delta = 0.3/mm
gamma = 20.
kappa = 1. + delta*mm
theta = np.array([alpha, beta, delta, gamma, kappa])


# Update definition with new theta
def pot_value(xx):
    grad = np.zeros(mm)
    wksp = np.zeros(mm)
    value = pot(xx, grad, orig, cost_mat, theta, nn, mm, wksp)
    return (value, grad)


# Run plots
plot_n = 100
space0 = -4.
space1 = .5
space = np.linspace(space0,space1, plot_n)
xx, yy = np.meshgrid(space, space)
zz = np.zeros((plot_n, plot_n))

alpha_values = np.array([.5, 1., 1.5, 2.])
plt.figure(figsize=(12,3))
for k in range(4):
    plt.subplot(1, 4, k+1)
    theta[0] = alpha_values[k]

    for i in range(plot_n):
        for j in range(plot_n):
            temp = np.array([xx[i, j], yy[i, j]])
            zz[i, j] = np.exp(-pot_value(temp)[0])

    plt.contourf(xx, yy, zz, 300)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([space0, space1])
    plt.ylim([space0, space1])
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()