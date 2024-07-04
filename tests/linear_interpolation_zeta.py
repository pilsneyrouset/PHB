import numpy as np
import unittest
from numpy.testing import assert_almost_equal
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions import linear_interpolation_zeta, interpolation

class TestInterpolationZeta(unittest.TestCase):
    def setUp(self):
        self.m = 1000  # number of tests
        self.alpha = 0.9
        self.K = 600  # length of the template
        self.s = 800  # length of the region we are interested in
        self.thresholds = np.array([self.alpha * k / self.m for k in range(self.K)])
    
    def test_constant_pvalues(self):
        zeta = [k for k in range(self.K)]
        p_values = [0] * self.s
        self.assertEqual(assert_almost_equal(linear_interpolation_zeta(p_values, self.thresholds, zeta=zeta), [interpolation(np.sort(p_values)[:i], self.thresholds, zeta=zeta) for i in range(1, self.s+1)]), None)

        zeta = [k**2 for k in range(self.K)]
        p_values = [1] * self.s
        self.assertEqual(assert_almost_equal(linear_interpolation_zeta(p_values, self.thresholds, zeta=zeta), [interpolation(np.sort(p_values)[:i], self.thresholds, zeta=zeta) for i in range(1, self.s+1)]), None)

        zeta = [k**2 + 3 for k in range(self.K)]
        p_values = [0.3] * self.s
        self.assertEqual(assert_almost_equal(linear_interpolation_zeta(p_values, self.thresholds, zeta=zeta), [interpolation(np.sort(p_values)[:i], self.thresholds, zeta=zeta) for i in range(1, self.s+1)]), None)
    
    def test_random_pvalues(self):
        zeta = [k for k in range(self.K)]
        p_values = np.random.uniform(low=0, high=1, size=self.s)
        self.assertEqual(assert_almost_equal(linear_interpolation_zeta(p_values, self.thresholds, zeta=zeta), [interpolation(np.sort(p_values)[:i], self.thresholds, zeta=zeta) for i in range(1, self.s+1)]), None)

        zeta = [k**2 for k in range(self.K)]
        p_values = np.random.beta(a=self.K, b=self.m - self.K + 1, size=self.s)
        self.assertEqual(assert_almost_equal(linear_interpolation_zeta(p_values, self.thresholds, zeta=zeta), [interpolation(np.sort(p_values)[:i], self.thresholds, zeta=zeta) for i in range(1, self.s+1)]), None)


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestInterpolationZeta)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)