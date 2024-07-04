
import numpy as np
from numpy.testing import assert_almost_equal
from sanssouci import curve_max_fp

from functions import interpolation

m = 10000
alpha = 0.9
K = 600
s = 800
np.random.seed(42)
values = np.linspace(0, 1, s)
p_values = values ** 2
p_values = [0]*s
p_values = [np.random.beta(a=k, b=m-k+1) for k in range(1, s+1)]
thresholds = np.array([alpha*k/m +0.01 for k in range(K)])
# thresholds = np.sort(np.random.uniform(low=0, high=1, size=K))
zeta = [k**2 for k in range(K)]


def linear_interpolation_zeta(p_values: list, thresholds: list, zeta: list, kmin: int=0) -> list:
    p_values = np.sort(p_values)
    s = len(p_values)
    K = len(thresholds)
    tau = np.zeros(K)
    ksi = np.zeros(K)
    K = max(s, K)
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
    V, A, M = np.zeros(s), np.zeros(K), np.zeros(K)
    M[0] = r[0]
    for k in range(K):
        A[k] = r[k] - ksi[k]
        if k > 0:
            M[k] = max(M[k-1], A[k])
    for i in range(s):
        if kappa[i] > 0:
            V[i] = int(min(ksi[kappa[i]], i+1 - M[kappa[i]-1]))
    return V