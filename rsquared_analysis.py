"""
R2 analysis for deterministic model defined in terms of potential function.
"""

from urban_model import *
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Initialize search grid
grid_n = 100
alpha_values = np.linspace(0, 2.0, grid_n+1)[1:]
beta_values = np.linspace(0, 1.4e6, grid_n+1)[1:]
XX, YY = np.meshgrid(alpha_values, beta_values)
r2_values = np.zeros((grid_n, grid_n))


# Search values
last_r2 = -np.infty
max_pot = -np.infty


# Total sum squares
w_data = np.exp(xd)
w_data_centred = w_data - np.mean(w_data)
ss_tot = np.dot(w_data_centred, w_data_centred)


# Perform grid evaluations
for i in range(grid_n):
    for j in range(grid_n):
        print("Running for " + str(i) + ", " + str(j))
        try:
            # Residiual sum squares
            theta[0] = XX[i, j]
            theta[1] = YY[i, j]
            w_pred = np.exp(minimize(pot_value, xd, method='L-BFGS-B', jac=True, options={'disp': False}).x)
            res = w_pred - w_data
            ss_res = np.dot(res, res)

            # Regression sum squares
            r2_values[i, j] = 1. - ss_res/ss_tot

        except:
            None

        # If minimize fails set value to previous, otherwise update previous
        if r2_values[i, j] == 0:
            r2_values[i, j] = last_r2
        else:
            last_r2 = r2_values[i, j]

# Output results
idx = np.unravel_index(r2_values.argmax(), r2_values.shape)
print("Fitted alpha and beta values:")
print(XX[idx], YY[idx]*2./1.4e6, r2_values[idx])
np.savetxt("output/rsquared_analysis.txt", r2_values)
plt.pcolor(XX, YY*2./1.4e6, r2_values)
plt.xlim([np.min(XX), np.max(XX)])
plt.ylim([np.min(YY)*2./1.4e6, np.max(YY)*2./1.4e6])
plt.colorbar()
plt.show()