"""Tests for per-gene (per-feature) sigma support."""

import warnings
import pytest
import jax
import jax.numpy as jnp
import mellon


@pytest.fixture
def multi_output_data():
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    n, d, p = 50, 2, 3
    X = jax.random.normal(k1, (n, d))
    y = jax.random.normal(k2, (n, p))
    sigma = jnp.array([0.5, 1.0, 2.0])
    return X, y, sigma


def _fit_per_gene(X, y, sigma, n_landmarks, obs_variance=False):
    est = mellon.FunctionEstimator(sigma=sigma, n_landmarks=n_landmarks, obs_variance=obs_variance)
    est.fit(X, y)
    return est


def _fit_scalar(X, y_col, sigma_val, n_landmarks, obs_variance=False):
    est = mellon.FunctionEstimator(sigma=sigma_val, n_landmarks=n_landmarks, obs_variance=obs_variance)
    est.fit(X, y_col)
    return est


@pytest.mark.parametrize("n_landmarks", [0, 15])
def test_pergene_matches_per_column_scalar(multi_output_data, n_landmarks):
    """Per-gene sigma predictions match gene-by-gene scalar fits."""
    X, y, sigma = multi_output_data
    p = y.shape[1]

    est_pg = _fit_per_gene(X, y, sigma, n_landmarks)
    pred_pg = est_pg.predict(X)

    for g in range(p):
        est_g = _fit_scalar(X, y[:, g], float(sigma[g]), n_landmarks)
        pred_g = est_g.predict(X)
        assert jnp.allclose(pred_pg[:, g], pred_g, atol=1e-5), (
            f"Gene {g}: per-gene pred != scalar pred (n_landmarks={n_landmarks}), "
            f"max diff = {jnp.max(jnp.abs(pred_pg[:, g] - pred_g))}"
        )


@pytest.mark.parametrize("n_landmarks", [0, 15])
def test_pergene_leverage_matches_per_column(multi_output_data, n_landmarks):
    """Per-gene leverage shape (n, p) matches per-column scalar computation."""
    X, y, sigma = multi_output_data
    p = y.shape[1]
    n = X.shape[0]

    est_pg = _fit_per_gene(X, y, sigma, n_landmarks)
    lev_pg = est_pg.predict.leverage(X)

    assert lev_pg.shape == (n, p), f"Expected ({n}, {p}), got {lev_pg.shape}"

    for g in range(p):
        est_g = _fit_scalar(X, y[:, g], float(sigma[g]), n_landmarks)
        lev_g = est_g.predict.leverage(X)
        assert jnp.allclose(lev_pg[:, g], lev_g, atol=1e-5), (
            f"Gene {g}: per-gene lev != scalar lev (n_landmarks={n_landmarks}), "
            f"max diff = {jnp.max(jnp.abs(lev_pg[:, g] - lev_g))}"
        )


@pytest.mark.parametrize("n_landmarks", [0, 15])
def test_pergene_obs_variance_shape(multi_output_data, n_landmarks):
    """Per-gene obs_variance has shape (n, p) and is mostly positive."""
    X, y, sigma = multi_output_data
    n, p = y.shape

    est = _fit_per_gene(X, y, sigma, n_landmarks, obs_variance=True)
    var = est.predict.obs_variance(X)

    assert var.shape == (n, p), f"Expected ({n}, {p}), got {var.shape}"
    assert jnp.mean(var > 0) > 0.9, "Most obs_variance values should be positive."


@pytest.mark.parametrize("n_landmarks", [0, 15])
def test_pergene_obs_variance_matches_per_column(multi_output_data, n_landmarks):
    """Per-gene obs_variance matches per-column scalar fits."""
    X, y, sigma = multi_output_data
    p = y.shape[1]

    est_pg = _fit_per_gene(X, y, sigma, n_landmarks, obs_variance=True)
    obsvar_pg = est_pg.predict.obs_variance(X)

    for g in range(p):
        est_g = _fit_scalar(X, y[:, g], float(sigma[g]), n_landmarks, obs_variance=True)
        obsvar_g = est_g.predict.obs_variance(X)
        assert jnp.allclose(obsvar_pg[:, g], obsvar_g, atol=1e-5), (
            f"Gene {g}: per-gene obsvar != scalar obsvar (n_landmarks={n_landmarks}), "
            f"max diff = {jnp.max(jnp.abs(obsvar_pg[:, g] - obsvar_g))}"
        )


@pytest.mark.parametrize("n_landmarks", [0, 15])
def test_pergene_loo_residuals_squared_matches_per_column(multi_output_data, n_landmarks):
    """Per-gene loo_residuals_squared matches per-column scalar computation."""
    X, y, sigma = multi_output_data
    n, p = y.shape

    est_pg = _fit_per_gene(X, y, sigma, n_landmarks)
    var_pg = est_pg.predict.loo_residuals_squared(X, y)

    assert var_pg.shape == (n, p), f"Expected ({n}, {p}), got {var_pg.shape}"
    assert jnp.all(var_pg >= 0), "Variance should be non-negative."

    for g in range(p):
        est_g = _fit_scalar(X, y[:, g], float(sigma[g]), n_landmarks)
        var_g = est_g.predict.loo_residuals_squared(X, y[:, g])
        assert jnp.allclose(var_pg[:, g], var_g, atol=1e-5), (
            f"Gene {g}: per-gene empvar != scalar empvar (n_landmarks={n_landmarks}), "
            f"max diff = {jnp.max(jnp.abs(var_pg[:, g] - var_g))}"
        )


@pytest.mark.parametrize("n_landmarks", [0, 15])
def test_pergene_leverage_range(multi_output_data, n_landmarks):
    """Per-feature leverage values should be in [0, 1)."""
    X, y, sigma = multi_output_data

    est = _fit_per_gene(X, y, sigma, n_landmarks)
    lev = est.predict.leverage(X)

    assert jnp.all(lev >= 0), "Negative leverage found with per-feature sigma."
    assert jnp.all(lev < 1), "Leverage >= 1 found with per-feature sigma."


def test_scalar_sigma_still_works():
    """Scalar sigma regression: unchanged behavior."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    X = jax.random.normal(k1, (30, 2))
    y = jax.random.normal(k2, (30, 3))

    est = mellon.FunctionEstimator(sigma=1.0, n_landmarks=0, obs_variance=True)
    est.fit(X, y)

    pred = est.predict(X)
    assert pred.shape == (30, 3)

    lev = est.predict.leverage(X)
    assert lev.shape == (30,)

    var = est.predict.obs_variance(X)
    assert var.shape == (30, 3)


def test_ambiguous_n_equals_p():
    """When n == p, 1D sigma is detected as per-feature and fitting works."""
    from mellon.conditional import _is_per_feature_sigma

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    n = 20
    X = jax.random.normal(k1, (n, 2))
    y = jax.random.normal(k2, (n, n))  # n == p
    sigma = jnp.ones(n) * 0.5

    # Detection helper treats it as per-feature
    assert _is_per_feature_sigma(sigma, y) is True

    # Fitting works and produces correct shape
    est = mellon.FunctionEstimator(sigma=sigma, n_landmarks=0)
    est.fit(X, y)
    pred = est.predict(X)
    assert pred.shape == (n, n)


def test_explicit_1p_shape():
    """Explicit (1, p) sigma forces per-feature interpretation."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    n, p = 30, 3
    X = jax.random.normal(k1, (n, 2))
    y = jax.random.normal(k2, (n, p))
    sigma = jnp.array([[0.5, 1.0, 2.0]])  # (1, 3)

    est = mellon.FunctionEstimator(sigma=sigma, n_landmarks=0)
    est.fit(X, y)

    pred = est.predict(X)
    assert pred.shape == (n, p)

    # Should give same result as (p,) sigma
    sigma_flat = jnp.array([0.5, 1.0, 2.0])
    est2 = mellon.FunctionEstimator(sigma=sigma_flat, n_landmarks=0)
    est2.fit(X, y)
    pred2 = est2.predict(X)

    assert jnp.allclose(pred, pred2, atol=1e-10)


def test_n1_shape_not_per_feature():
    """Shape (n, 1) should NOT be treated as per-feature (reserved for future per-obs)."""
    from mellon.conditional import _is_per_feature_sigma

    n, p = 30, 3
    sigma = jnp.ones((n, 1))
    y = jnp.ones((n, p))

    assert not _is_per_feature_sigma(sigma, y)


def test_estimator_convenience_with_pergene(multi_output_data):
    """estimator.leverage() and estimator.loo_residuals_squared() work with per-gene sigma."""
    X, y, sigma = multi_output_data

    est = mellon.FunctionEstimator(sigma=sigma, n_landmarks=0)
    est.fit(X, y)

    lev = est.leverage(X)
    assert lev.shape == (X.shape[0], y.shape[1])

    var = est.loo_residuals_squared(X, y)
    assert var.shape == y.shape


def test_pergene_cached_loo_matches_explicit(multi_output_data):
    """Per-feature sigma + obs_variance: cached loo matches explicit."""
    X, y, sigma = multi_output_data

    est = mellon.FunctionEstimator(sigma=sigma, n_landmarks=15, obs_variance=True)
    est.fit(X, y)

    var_cached = est.loo_residuals_squared()
    var_explicit = est.predict.loo_residuals_squared(X, y)

    assert jnp.allclose(var_cached, var_explicit, atol=1e-3), (
        "Cached per-gene loo should match explicit computation."
    )
