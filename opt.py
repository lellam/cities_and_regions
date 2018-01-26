"""
Find global minimum of p(x | theta)
"""

from urban_model import *
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Set theta for low-noise model
theta[0] = 0.5
theta[1] = 0.3*0.7e6
theta[2] = 0.3/mm
theta[3] = 10000.
theta[4] = 1.3


# Run optimization
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

# Save down
np.savetxt("output/opt" + str(theta[0]) + ".txt", m)
