"""Tests for per-observation-per-feature (n, p) sigma support."""

import pytest
import jax
import jax.numpy as jnp
import mellon


@pytest.fixture
def multi_output_data():
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    n, d, p = 50, 2, 3
    X = jax.random.normal(k1, (n, d))
    y = jax.random.normal(k2, (n, p))
    # Per-observation-per-feature sigma: (n, p)
    sigma = jax.random.uniform(k3, (n, p), minval=0.3, maxval=2.0)
    return X, y, sigma


def _fit_np_sigma(X, y, sigma, n_landmarks, with_uncertainty=False):
    est = mellon.FunctionEstimator(
        sigma=sigma, n_landmarks=n_landmarks,
        predictor_with_uncertainty=with_uncertainty,
    )
    est.fit(X, y)
    return est


def _fit_scalar(X, y_col, sigma_col, n_landmarks, with_uncertainty=False):
    est = mellon.FunctionEstimator(
        sigma=sigma_col, n_landmarks=n_landmarks,
        predictor_with_uncertainty=with_uncertainty,
    )
    est.fit(X, y_col)
    return est


@pytest.mark.parametrize("n_landmarks", [0, 15])
def test_np_sigma_predictions_match_per_column(multi_output_data, n_landmarks):
    """(n, p) sigma predictions match fitting each column separately."""
    X, y, sigma = multi_output_data
    p = y.shape[1]

    est_np = _fit_np_sigma(X, y, sigma, n_landmarks)
    pred_np = est_np.predict(X)

    for g in range(p):
        est_g = _fit_scalar(X, y[:, g], sigma[:, g], n_landmarks)
        pred_g = est_g.predict(X)
        assert jnp.allclose(pred_np[:, g], pred_g, atol=1e-5), (
            f"Gene {g}: (n,p) sigma pred != scalar pred (n_landmarks={n_landmarks}), "
            f"max diff = {jnp.max(jnp.abs(pred_np[:, g] - pred_g))}"
        )


@pytest.mark.parametrize("n_landmarks", [15])
def test_np_sigma_covariance_is_sigma_independent(multi_output_data, n_landmarks):
    """GP posterior covariance depends only on data locations, not on sigma.

    The posterior covariance k_post(x,x') = k(x,x') - K_{x,X} K_{X,X}^{-1} K_{X,x'}
    captures epistemic uncertainty from data point availability. Observation noise
    is accounted for separately via obs_variance. Therefore the covariance should
    be the same regardless of sigma, and should return shape (n_test,) not (n_test, p).
    """
    X, y, sigma = multi_output_data
    n_test = 10
    X_test = X[:n_test]

    # Fit with (n, p) sigma
    est_np = _fit_np_sigma(X, y, sigma, n_landmarks, with_uncertainty=True)
    cov_np = est_np.predict.covariance(X_test, diag=True)

    # Covariance is shared across features — shape (n_test,)
    assert cov_np.shape == (n_test,), (
        f"Expected covariance shape ({n_test},), got {cov_np.shape}"
    )

    # Should match a scalar sigma fit — covariance structure is sigma-independent
    est_scalar = mellon.FunctionEstimator(
        sigma=1.0, n_landmarks=n_landmarks, predictor_with_uncertainty=True,
    )
    est_scalar.fit(X, y)
    cov_scalar = est_scalar.predict.covariance(X_test, diag=True)

    assert jnp.allclose(cov_np, cov_scalar, atol=1e-4), (
        f"Per-feature sigma covariance should match scalar sigma covariance, "
        f"max diff = {jnp.max(jnp.abs(cov_np - cov_scalar))}"
    )


def test_np_sigma_detected():
    """(n, p) sigma is detected as per-feature by _is_per_feature_sigma."""
    from mellon.conditional import _is_per_feature_sigma

    n, p = 50, 3
    sigma = jnp.ones((n, p))
    y = jnp.ones((n, p))
    assert _is_per_feature_sigma(sigma, y) is True


def test_np_sigma_not_confused_with_n1():
    """(n, 1) sigma should NOT be treated as per-feature."""
    from mellon.conditional import _is_per_feature_sigma

    n, p = 50, 3
    sigma = jnp.ones((n, 1))
    y = jnp.ones((n, p))
    assert _is_per_feature_sigma(sigma, y) is False


@pytest.mark.parametrize("n_landmarks", [15])
def test_np_sigma_covariance_positive(multi_output_data, n_landmarks):
    """Covariance values should be positive."""
    X, y, sigma = multi_output_data

    est = _fit_np_sigma(X, y, sigma, n_landmarks, with_uncertainty=True)
    cov = est.predict.covariance(X[:10], diag=True)

    assert jnp.all(cov > 0), "Covariance should be positive."


@pytest.mark.parametrize("n_landmarks", [15])
def test_np_sigma_full_covariance(multi_output_data, n_landmarks):
    """Full covariance matrix (diag=False) works with per-feature sigma."""
    X, y, sigma = multi_output_data
    n_test = 5
    X_test = X[:n_test]

    est = _fit_np_sigma(X, y, sigma, n_landmarks, with_uncertainty=True)
    cov = est.predict.covariance(X_test, diag=False)

    assert cov.shape == (n_test, n_test), (
        f"Expected shape ({n_test}, {n_test}), got {cov.shape}"
    )
