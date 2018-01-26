"""
To generate random stopping times from P(K > k) = 1./k^1.1
"""

import numpy as np
import matplotlib.pyplot as plt

nums = np.empty(20000)

for i in range(20000):
    N = 1
    k_pow = 1.1
    u = np.random.uniform(0, 1)
    while(u < np.power(N+1, -k_pow)):
        N += 1
    nums[i] = N

np.savetxt("stopping.txt", nums)
