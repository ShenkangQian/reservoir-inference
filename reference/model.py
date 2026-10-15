"""Fixed-scale inference for a one-layer random-feature regressor."""

from dataclasses import dataclass

import numpy as np


def features(x, a, b):
    """Return the n-by-k tanh feature matrix for scalar inputs."""
    return np.tanh(np.asarray(x).reshape(-1, 1) * np.asarray(a) + np.asarray(b))


def readout_posterior(f, y, penalty, noise_variance):
    """Gaussian posterior for w with prior variance noise_variance / penalty."""
    f, y = np.asarray(f, dtype=float), np.asarray(y, dtype=float)
    if f.ndim != 2 or y.shape != (len(f),):
        raise ValueError("Expected a feature matrix and one response per row.")
    if penalty <= 0 or noise_variance <= 0:
        raise ValueError("The penalty and noise variance must be positive.")
    precision_factor = np.linalg.cholesky(f.T @ f + penalty * np.eye(f.shape[1]))
    mean = np.linalg.solve(precision_factor.T, np.linalg.solve(precision_factor, f.T @ y))
    inv_factor = np.linalg.solve(precision_factor, np.eye(f.shape[1]))
    covariance = noise_variance * inv_factor.T @ inv_factor
    return mean, covariance


@dataclass
class DiagonalHiddenPosterior:
    mean_a: np.ndarray
    mean_b: np.ndarray
    variance_a: np.ndarray
    variance_b: np.ndarray

    def sample(self, rng, draws):
        return (
            rng.normal(self.mean_a, np.sqrt(self.variance_a), (draws, len(self.mean_a))),
            rng.normal(self.mean_b, np.sqrt(self.variance_b), (draws, len(self.mean_b))),
        )


def diagonal_tagi(x, y, fixed_readout, noise_variance, variance_a, variance_b):
    """One sequential pass, starting from independent zero-mean Gaussian priors.

    The fixed readout is kept constant throughout the pass. All coordinates in
    an observation update use the same pre-update moments.
    """
    x, y, w = np.asarray(x), np.asarray(y), np.asarray(fixed_readout)
    if x.ndim != 1 or x.shape != y.shape or w.ndim != 1:
        raise ValueError("Inputs, responses, and readout must be vectors.")
    if min(noise_variance, variance_a, variance_b) <= 0:
        raise ValueError("All prior and noise variances must be positive.")
    ma, mb = np.zeros_like(w, dtype=float), np.zeros_like(w, dtype=float)
    va, vb = np.full_like(ma, variance_a), np.full_like(mb, variance_b)
    for xi, yi in zip(x, y):
        activation = np.tanh(xi * ma + mb)
        weighted_jacobian = w * (1.0 - activation**2)
        predictive_variance = noise_variance + np.sum(
            weighted_jacobian**2 * (xi**2 * va + vb)
        )
        residual = yi - w @ activation
        ca, cb = weighted_jacobian * xi * va, weighted_jacobian * vb
        ma = ma + ca * residual / predictive_variance
        mb = mb + cb * residual / predictive_variance
        va = np.maximum(va - ca**2 / predictive_variance, 0.0)
        vb = np.maximum(vb - cb**2 / predictive_variance, 0.0)
    return DiagonalHiddenPosterior(ma, mb, va, vb)


def function_draws(x, y, grid, hidden, penalty, noise_variance, rng, draws=200):
    """Paired ridge-only and conditional-readout draws at the same hidden draws."""
    a, b = hidden.sample(rng, draws)
    ridge, conditional = np.empty((draws, len(grid))), np.empty((draws, len(grid)))
    for i in range(draws):
        f, fs = features(x, a[i], b[i]), features(grid, a[i], b[i])
        mean, covariance = readout_posterior(f, y, penalty, noise_variance)
        ridge[i] = fs @ mean
        conditional[i] = fs @ rng.multivariate_normal(mean, covariance)
    return ridge, conditional
