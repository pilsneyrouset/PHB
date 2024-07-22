import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions import linear_interpolation, interpolation, interpolation_minmax

class TestLinearInterpolation(unittest.TestCase):
    def setUp(self):
        self.m = 1000  # number of tests
        self.alpha = 0.9
        self.K = 700  # length of the template
        self.s = 600  # length of the region we are interested in
        self.thresholds = np.array([self.alpha * k / self.m +0.01 for k in range(self.K)])
        self.kmin = 0
        self.zeta = [k for k in range(self.K)]
    
    def test_constant_pvalues(self):
        p_values = [0] * self.s
        self.assertEqual(assert_almost_equal(linear_interpolation(p_values, self.thresholds),[interpolation(np.sort(p_values)[:i], self.thresholds, zeta=self.zeta) for i in range(1, self.s+1)] ), None)
        self.assertEqual(linear_interpolation(p_values, self.thresholds)[-1], interpolation_minmax(p_values, self.thresholds, kmin=self.kmin, kmax=self.K, zeta=self.zeta))

        p_values = [1] * self.s
        self.assertEqual(assert_almost_equal(linear_interpolation(p_values, self.thresholds),[interpolation(np.sort(p_values)[:i], self.thresholds, zeta=self.zeta) for i in range(1, self.s+1)] ), None)
        self.assertEqual(linear_interpolation(p_values, self.thresholds)[-1], interpolation_minmax(p_values, self.thresholds, kmin=self.kmin, kmax=self.K, zeta=self.zeta))

        p_values = [0.3] * self.s
        self.assertEqual(assert_almost_equal(linear_interpolation(p_values, self.thresholds),[interpolation(np.sort(p_values)[:i], self.thresholds, zeta=self.zeta) for i in range(1, self.s+1)] ), None)
        self.assertEqual(linear_interpolation(p_values, self.thresholds)[-1], interpolation_minmax(p_values, self.thresholds, kmin=self.kmin, kmax=self.K, zeta=self.zeta))
    
    def test_random_pvalues(self):
        p_values = np.random.uniform(low=0, high=1, size=self.s)
        self.assertEqual(assert_almost_equal(linear_interpolation(p_values, self.thresholds),[interpolation(np.sort(p_values)[:i], self.thresholds, zeta=self.zeta) for i in range(1, self.s+1)] ), None)
        self.assertEqual(linear_interpolation(p_values, self.thresholds)[-1], interpolation_minmax(p_values, self.thresholds, kmin=self.kmin, kmax=self.K, zeta=self.zeta))

        p_values = np.random.beta(a=self.K, b=self.m - self.K + 1, size=self.s)
        self.assertEqual(assert_almost_equal(linear_interpolation(p_values, self.thresholds),[interpolation(np.sort(p_values)[:i], self.thresholds, zeta=self.zeta) for i in range(1, self.s+1)] ), None)
        self.assertEqual(linear_interpolation(p_values, self.thresholds)[-1], interpolation_minmax(p_values, self.thresholds, kmin=self.kmin, kmax=self.K, zeta=self.zeta))


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestLinearInterpolation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)