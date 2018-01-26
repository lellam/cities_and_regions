"""
Just some wrappers...
"""

import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer


# Load shared object
lib = ctypes.cdll.LoadLibrary("./potential_functions.so")

# Load pot function from shared object
pot = lib.potential
pot.restype = ctypes.c_double
pot.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

# Load hess function from shared object
hess = lib.hessian
hess.restype = None
hess.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

# Declare global vars
cost_mat = np.loadtxt("data/london_n/cost_mat.txt")
orig = np.loadtxt("data/london_n/P.txt")
xd = np.loadtxt("data/london_n/xd0.txt")
nn, mm = np.shape(cost_mat)
theta = np.array([1., 0., .3/mm, 100., 1.3])

# Wrapper for potential function
def pot_value(xx):
    grad = np.zeros(mm)
    wksp = np.zeros(mm)
    value = pot(xx, grad, orig, cost_mat, theta, nn, mm, wksp)
    return (value, grad)

# Wrapper for hessian function
def pot_hess(xx):
    A = np.zeros((mm, mm))
    wksp = np.zeros(mm)
    hess(xx, A, orig, cost_mat, theta, nn, mm, wksp)
    return A

# Potential function of the likelihood
s2_inv = 100.
def like_value(xx):
    diff = xx - xd
    grad = s2_inv*diff
    pot = 0.5*s2_inv*np.dot(diff, diff)
    return pot, grad

# Potential function for annealed importance sampling (no flows model)
def pot0_value(xx):
    delta = theta[2]
    gamm = theta[3]
    kk = theta[4]
    gamm_kk_exp_xx = gamm*kk*np.exp(xx)
    gradV = -gamm*(delta+1./mm)*np.ones(mm) + gamm_kk_exp_xx
    V = -gamm*(delta+1./mm)*xx.sum() + gamm_kk_exp_xx.sum()
    return V, gradV