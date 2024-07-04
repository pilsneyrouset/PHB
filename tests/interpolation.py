import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions import interpolation, interpolation_naif

class TestInterpolation(unittest.TestCase):
    def setUp(self):
        self.m = 1000  # number of tests
        self.alpha = 0.9
        self.K = 500  # length of the template
        self.s = 600  # length of the region we are interested in
        self.thresholds = np.array([self.alpha * k / self.m for k in range(self.K)])
    
    def test_constant_pvalues(self):
        p_values = [0] * self.s
        self.assertEqual(interpolation(p_values, self.thresholds), interpolation_naif(p_values, self.thresholds))

        p_values = [1] * self.s
        self.assertEqual(interpolation(p_values, self.thresholds), interpolation_naif(p_values, self.thresholds))

        p_values = [0.3] * self.s
        self.assertEqual(interpolation(p_values, self.thresholds), interpolation_naif(p_values, self.thresholds))
    
    def test_random_pvalues(self):
        p_values = np.random.uniform(low=0, high=1, size=self.s)
        self.assertEqual(interpolation(p_values, self.thresholds), interpolation_naif(p_values, self.thresholds))

        p_values = np.random.beta(a=self.K, b=self.m - self.K + 1, size=self.s)
        self.assertEqual(interpolation(p_values, self.thresholds), interpolation_naif(p_values, self.thresholds))


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestInterpolation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)