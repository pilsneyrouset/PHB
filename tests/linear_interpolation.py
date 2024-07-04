import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from functions.posthoc_bounds import linear_interpolation, interpolation, interpolation_minmax

class TestLinearInterpolation(unittest.TestCase):
    def setUp(self):
        self.m = 1000  # number of tests
        self.alpha = 0.9
        self.K = 700  # length of the template
        self.s = 600  # length of the region we are interested in
        self.thresholds = np.array([self.alpha * k / self.m for k in range(self.K)])
        self.kmin = np.random.randint(low=1, high=self.K)
    
    def test_constant_pvalues(self):
        p_values = [0] * self.s
        self.assertEqual(assert_almost_equal(linear_interpolation(p_values, self.thresholds),[interpolation(np.sort(p_values)[:i], self.thresholds) for i in range(1, self.s+1)] ), None)
        self.assertEqual(linear_interpolation(p_values, self.thresholds, kmin=self.kmin)[-1], interpolation_minmax(p_values, self.thresholds, kmin=self.kmin, kmax=self.K))

        p_values = [1] * self.s
        self.assertEqual(assert_almost_equal(linear_interpolation(p_values, self.thresholds),[interpolation(np.sort(p_values)[:i], self.thresholds) for i in range(1, self.s+1)] ), None)
        self.assertEqual(linear_interpolation(p_values, self.thresholds, kmin=self.kmin)[-1], interpolation_minmax(p_values, self.thresholds, kmin=self.kmin, kmax=self.K))

        p_values = [0.3] * self.s
        self.assertEqual(assert_almost_equal(linear_interpolation(p_values, self.thresholds),[interpolation(np.sort(p_values)[:i], self.thresholds) for i in range(1, self.s+1)] ), None)
        self.assertEqual(linear_interpolation(p_values, self.thresholds, kmin=self.kmin)[-1], interpolation_minmax(p_values, self.thresholds, kmin=self.kmin, kmax=self.K))
    
    def test_random_pvalues(self):
        p_values = np.random.uniform(low=0, high=1, size=self.s)
        self.assertEqual(assert_almost_equal(linear_interpolation(p_values, self.thresholds),[interpolation(np.sort(p_values)[:i], self.thresholds) for i in range(1, self.s+1)] ), None)
        self.assertEqual(linear_interpolation(p_values, self.thresholds, kmin=self.kmin)[-1], interpolation_minmax(p_values, self.thresholds, kmin=self.kmin, kmax=self.K))

        p_values = np.random.beta(a=self.K, b=self.m - self.K + 1, size=self.s)
        self.assertEqual(assert_almost_equal(linear_interpolation(p_values, self.thresholds),[interpolation(np.sort(p_values)[:i], self.thresholds) for i in range(1, self.s+1)] ), None)
        self.assertEqual(linear_interpolation(p_values, self.thresholds, kmin=self.kmin)[-1], interpolation_minmax(p_values, self.thresholds, kmin=self.kmin, kmax=self.K))

def run_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestLinearInterpolation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result

run_tests()

if __name__ == '__main__':
    unittest.main()