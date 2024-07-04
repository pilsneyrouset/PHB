#%%
import numpy as np
from numpy.testing import assert_almost_equal
from sanssouci import curve_max_fp

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
zeta = [k**2 for k in range(5000)]