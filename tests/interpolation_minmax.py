import unittest
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions import interpolation_minmax, interpolation_minmax_naif

class TestInterpolationMinmax(unittest.TestCase):
    def setUp(self):
        self.m = 1000  # number of tests
        self.alpha = 0.9
        self.K = 500  # length of the template
        self.s = 600  # length of the region we are interested in
        self.thresholds = np.array([self.alpha * k / self.m for k in range(self.K)])
        self.kmin = np.random.randint(low=0, high=self.K)
        self.kmax = np.random.randint(low=self.kmin, high=self.K)
        self.zeta = [k for k in range(self.K)]
    
    def test_constant_pvalues(self):
        p_values = [0] * self.s
        self.assertEqual(interpolation_minmax(p_values, self.thresholds, kmin=self.kmin, kmax=self.kmax, zeta=self.zeta), interpolation_minmax_naif(p_values, self.thresholds, kmin=self.kmin, kmax=self.kmax, zeta=self.zeta))

        p_values = [1] * self.s
        self.assertEqual(interpolation_minmax(p_values, self.thresholds, kmin=self.kmin, kmax=self.kmax, zeta=self.zeta), interpolation_minmax_naif(p_values, self.thresholds, kmin=self.kmin, kmax=self.kmax, zeta=self.zeta))

        p_values = [0.3] * self.s
        self.assertEqual(interpolation_minmax(p_values, self.thresholds, kmin=self.kmin, kmax=self.kmax, zeta=self.zeta), interpolation_minmax_naif(p_values, self.thresholds, kmin=self.kmin, kmax=self.kmax, zeta=self.zeta))
    def test_random_pvalues(self):
        p_values = np.random.uniform(low=0, high=1, size=self.s)
        self.assertEqual(interpolation_minmax(p_values, self.thresholds, kmin=self.kmin, kmax=self.kmax, zeta=self.zeta), interpolation_minmax_naif(p_values, self.thresholds, kmin=self.kmin, kmax=self.kmax, zeta=self.zeta))

        p_values = np.random.beta(a=self.K, b=self.m - self.K + 1, size=self.s)
        self.assertEqual(interpolation_minmax(p_values, self.thresholds, kmin=self.kmin, kmax=self.kmax, zeta=self.zeta), interpolation_minmax_naif(p_values, self.thresholds, kmin=self.kmin, kmax=self.kmax, zeta=self.zeta))
    
    def test_indices(self):
        p_values = np.random.beta(a=self.K, b=self.m - self.K + 1, size=self.s)
        self.assertEqual(interpolation_minmax(p_values, self.thresholds, kmin=self.K, kmax=self.K, zeta=self.zeta), interpolation_minmax_naif(p_values, self.thresholds, kmin=self.K, kmax=self.K, zeta=self.zeta))
        self.assertEqual(interpolation_minmax(p_values, self.thresholds, kmin=0, kmax=0, zeta=self.zeta), interpolation_minmax_naif(p_values, self.thresholds, kmin=0, kmax=0, zeta=self.zeta))


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestInterpolationMinmax)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)