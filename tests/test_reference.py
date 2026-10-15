import unittest

import numpy as np

from reference.model import diagonal_tagi, features, readout_posterior


class ReferenceIdentities(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(17)

    def test_readout_normal_equations_and_covariance(self):
        f = self.rng.normal(size=(40, 6))
        y = self.rng.normal(size=40)
        mean, covariance = readout_posterior(f, y, 0.3, 0.2)
        q = f.T @ f + 0.3 * np.eye(6)
        np.testing.assert_allclose(q @ mean, f.T @ y, atol=1e-12)
        np.testing.assert_allclose(q @ covariance, 0.2 * np.eye(6), atol=1e-12)

    def test_sign_and_permutation_symmetry(self):
        x = np.linspace(-2, 2, 31)
        a, b, w = self.rng.normal(size=(3, 8))
        p = self.rng.permutation(8)
        s = self.rng.choice([-1, 1], 8)
        original = features(x, a, b) @ w
        transformed = features(x, s * a[p], s * b[p]) @ (s * w[p])
        np.testing.assert_allclose(original, transformed, atol=1e-12)

    def test_zero_readout_preserves_prior(self):
        p = diagonal_tagi(np.array([0., 1.]), np.array([2., 3.]), np.zeros(4), .1, .6, .7)
        np.testing.assert_array_equal(p.mean_a, np.zeros(4))
        np.testing.assert_array_equal(p.mean_b, np.zeros(4))
        np.testing.assert_array_equal(p.variance_a, np.full(4, .6))
        np.testing.assert_array_equal(p.variance_b, np.full(4, .7))

    def test_diagonal_variances_contract(self):
        x = np.linspace(-3, 3, 80)
        p = diagonal_tagi(x, np.sin(x), self.rng.normal(size=5), .1, .6, .7)
        self.assertTrue(np.all((p.variance_a >= 0) & (p.variance_a <= .6)))
        self.assertTrue(np.all((p.variance_b >= 0) & (p.variance_b <= .7)))
        self.assertTrue(np.isfinite(p.mean_a).all())

    def test_conditional_readout_adds_positive_function_variance(self):
        f = self.rng.normal(size=(20, 3))
        mean, covariance = readout_posterior(f, self.rng.normal(size=20), .5, .2)
        phi = np.array([.5, -.2, 1.0])
        expected = phi @ covariance @ phi
        samples = self.rng.multivariate_normal(mean, covariance, size=40000) @ phi
        self.assertGreater(expected, 0)
        self.assertAlmostEqual(samples.var() / expected, 1.0, delta=.04)


if __name__ == "__main__":
    unittest.main()
