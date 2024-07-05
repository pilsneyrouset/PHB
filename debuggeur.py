#%%
import numpy as np
from numpy.testing import assert_almost_equal
import sys
import os

from functions import interpolation

m = 1000
alpha = 0.05
K = 800
s = 800
np.random.seed(42)
values = np.linspace(0, 1, s)
pval = np.array([values ** 2, [0]*s, [1]*s, [np.random.beta(a=k, b=m-k+1) for k in range(1, s+1)]])
thresholds = np.array([alpha*k/m +0.01 for k in range(K)])
# thresholds = np.sort(np.random.uniform(low=0, high=1, size=K))
zetas = np.array([[k**2 for k in range(K)], [k for k in range(K)], [k**3 + 1 for k in range(K)]])

#%%
def linear_interpolation_zeta(p_values: list, thresholds: list, zeta: list, kmin: int=0) -> list:
    p_values = np.sort(p_values)
    s = len(p_values)
    K = len(thresholds)
    K = max(s, K)
    tau = np.zeros(K)
    ksi = np.zeros(K)
    for k in range(K):
        if k < kmin:
            tau[k] = 0
            ksi[k] = zeta[k]
        elif kmin <= k < len(thresholds):
            tau[k] = thresholds[k]
            ksi[k] = zeta[k]
        else:
            tau[k] = thresholds[len(thresholds)-1]
            ksi[k] = zeta[len(thresholds)-1]
    kappa = np.ones(K, dtype=int)*K
    r = np.ones(K, dtype=int)*s
    k, i = 0, 0
    while k < K and i < s:
        if p_values[i] < tau[k]:
            kappa[i] = k
            i += 1
        else:
            r[k] = i
            k += 1
    V, A, M = np.ones(s)*ksi[0], np.zeros(K), np.zeros(K)
    M[0] = max(r[0] - ksi[0], 0)
    for k in range(K):
        A[k] = r[k] - ksi[k]
        if k > 0:
            M[k] = max(M[k-1], A[k])
    for i in range(s):
        if kappa[i] > 0:
                if kappa[i] != K:
                    V[i] = int(min(ksi[kappa[i]], i+1 - M[kappa[i]-1]))
                else:
                    V[i] = int(i+1 - M[kappa[i]-1])
    return V
#%%
p_values = pval[2]
zeta = zetas[2]

a =linear_interpolation_zeta(p_values, thresholds, zeta=zeta)
b = [interpolation(np.sort(p_values)[:i], thresholds, zeta=zeta) for i in range(1, s+1)]
#%%
assert_almost_equal(a, b)
