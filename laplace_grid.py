"""
Evaluates p(x | theta) over grid of alpha and beta values
"""

from urban_model import *
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Set gamma value
theta[3] = 100


# To estimate z(theta) with Laplace approximation
def laplace_z():
    m = minimize(pot_value, xd0, method='L-BFGS-B', jac=True, options={'disp': False}).x
    A = pot_hess(m)
    L = np.linalg.cholesky(A)
    half_log_det_A = np.sum(np.log(np.diag(L)))
    return  -pot_value(m)[0] + lap_c1 -  half_log_det_A


# Initialize search grid
grid_n = 100
alpha_values = np.linspace(0, 2.0, grid_n+1)[1:]
beta_values = np.linspace(0, 1.4e6, grid_n+1)[1:]
XX, YY = np.meshgrid(alpha_values, beta_values)
like_values = np.zeros((grid_n, grid_n))
lap_c1 = 0.5*mm*np.log(2.*np.pi)


# Search values
last_like = -np.infty
max_pot = -np.infty


# Perform grid evaluations
for i in range(grid_n):
    for j in range(grid_n):
        print("Running for " + str(i) + ", " + str(j))

        theta[0] = XX[i, j]
        theta[1] = YY[i, j]
        try:
            # Run L-BFGS with mm different starts
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

            # Estimate likelihood with Laplace approximation
            A = pot_hess(m)
            L = np.linalg.cholesky(A)
            half_log_det_A = np.sum(np.log(np.diag(L)))
            lap =  -pot_value(m)[0] + lap_c1 -  half_log_det_A
            like_values[i, j] = -lap - pot_value(xd)[0]
        except:
            None
        
        # If minimize fails set value to previous, otherwise update previous
        if like_values[i, j] == 0:
            like_values[i, j] = last_like
        else:
            last_like = like_values[i, j]


# Output results
idx = np.unravel_index(like_values.argmax(), like_values.shape)
print("Fitted alpha and beta values:")
print(XX[idx], YY[idx]*2./1.4e6, like_values[idx])
np.savetxt("output/laplace_analysis" + str(theta[3]) + ".txt", like_values)
plt.pcolor(XX, YY*2./1.4e6, like_values)
plt.xlim([np.min(XX), np.max(XX)])
plt.ylim([np.min(YY)*2./1.4e6, np.max(YY)*2./1.4e6])
plt.colorbar()
plt.show()
