import pytest
import jax
import jax.numpy as jnp
import mellon


def _spearman_correlation(a, b):
    """Simple Spearman rank correlation without scipy."""
    a, b = jnp.asarray(a).ravel(), jnp.asarray(b).ravel()
    rank_a = jnp.argsort(jnp.argsort(a)).astype(float)
    rank_b = jnp.argsort(jnp.argsort(b)).astype(float)
    return jnp.corrcoef(rank_a, rank_b)[0, 1]


@pytest.fixture
def setup_data():
    n = 50
    d = 2
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    X = jax.random.normal(k1, (n, d))
    y = jnp.sum(jnp.sin(X), axis=1) + 0.1 * jax.random.normal(k2, (n,))
    return X, y


def test_full_gp_leverage_matches_explicit(setup_data):
    """Full GP: leverage matches explicit formula diag(K (K + sigma^2 I)^{-1})."""
    X, y = setup_data
    sigma = 1.0

    est = mellon.FunctionEstimator(sigma=sigma, n_landmarks=0)
    est.fit(X, y)

    h = est.predict.leverage(X, sigma=sigma)

    # Explicit computation
    K = est.predict.cov_func(X, X)
    n = X.shape[0]
    H = K @ jnp.linalg.inv(K + sigma**2 * jnp.eye(n))
    h_explicit = jnp.diag(H)

    assert jnp.allclose(h, h_explicit, atol=1e-4), (
        f"Leverage mismatch: max diff = {jnp.max(jnp.abs(h - h_explicit))}"
    )


def test_sparse_gp_leverage_correlates_with_full(setup_data):
    """Sparse GP leverage should correlate with full GP leverage."""
    X, y = setup_data
    sigma = 1.0

    est_full = mellon.FunctionEstimator(sigma=sigma, n_landmarks=0)
    est_full.fit(X, y)
    h_full = est_full.predict.leverage(X, sigma=sigma)

    est_sparse = mellon.FunctionEstimator(sigma=sigma, n_landmarks=20)
    est_sparse.fit(X, y)
    h_sparse = est_sparse.predict.leverage(X, sigma=sigma)

    corr = _spearman_correlation(h_full, h_sparse)
    assert corr > 0.8, f"Spearman correlation {corr} too low between full and sparse leverage."


def test_leverage_range(setup_data):
    """For any sigma > 0, all leverage values should be in [0, 1)."""
    X, y = setup_data

    for sigma in [0.5, 1.0, 2.0]:
        est = mellon.FunctionEstimator(sigma=sigma, n_landmarks=0)
        est.fit(X, y)
        h = est.predict.leverage(X, sigma=sigma)
        assert jnp.all(h >= 0), f"Negative leverage found for sigma={sigma}."
        assert jnp.all(h < 1), f"Leverage >= 1 found for sigma={sigma}."


def test_trace_bounded_by_m(setup_data):
    """sum(h) <= m (number of landmarks) for the sparse GP."""
    X, y = setup_data
    m = 15
    sigma = 1.0

    est = mellon.FunctionEstimator(sigma=sigma, n_landmarks=m)
    est.fit(X, y)
    h = est.predict.leverage(X, sigma=sigma)
    assert jnp.sum(h) <= m + 0.1, f"Trace {jnp.sum(h)} exceeds m={m}."


def test_sigma_dependence(setup_data):
    """Lower noise should give higher leverage."""
    X, y = setup_data

    est = mellon.FunctionEstimator(sigma=0.5, n_landmarks=0)
    est.fit(X, y)
    h_low = est.predict.leverage(X, sigma=0.5)

    est2 = mellon.FunctionEstimator(sigma=2.0, n_landmarks=0)
    est2.fit(X, y)
    h_high = est2.predict.leverage(X, sigma=2.0)

    assert jnp.mean(h_low) > jnp.mean(h_high), (
        f"Expected mean leverage at sigma=0.5 ({jnp.mean(h_low)}) > "
        f"sigma=2.0 ({jnp.mean(h_high)})"
    )


def test_estimator_convenience(setup_data):
    """estimator.leverage(X) should equal predictor.leverage(X, sigma=estimator.sigma)."""
    X, y = setup_data
    sigma = 1.0

    est = mellon.FunctionEstimator(sigma=sigma, n_landmarks=0)
    est.fit(X, y)

    h_convenience = est.leverage(X)
    h_direct = est.predict.leverage(X, sigma=sigma)

    assert jnp.allclose(h_convenience, h_direct), "Convenience method should match direct call."


def test_loo_residuals_squared(setup_data):
    """Empirical variance should be r^2 / (1 - h)^2."""
    X, y = setup_data
    sigma = 1.0

    est = mellon.FunctionEstimator(sigma=sigma, n_landmarks=0)
    est.fit(X, y)

    var = est.predict.loo_residuals_squared(X, y, sigma=sigma)

    prediction = est.predict(X)
    residual = y - prediction
    h = est.predict.leverage(X, sigma=sigma)
    expected = residual**2 / (1 - h) ** 2

    assert jnp.allclose(var, expected, atol=1e-6), "Empirical variance mismatch."


def test_loo_residuals_squared_multi_output():
    """Empirical variance should handle (n, p) multi-output observations."""
    n, d, p = 80, 3, 4
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    X = jax.random.normal(k1, (n, d))
    y = jax.random.normal(k2, (n, p))

    est = mellon.FunctionEstimator(n_landmarks=20, sigma=1.0)
    est.fit(X, y)

    var = est.predict.loo_residuals_squared(X, y, sigma=1.0)
    assert var.shape == (n, p), f"Expected ({n}, {p}), got {var.shape}"
    assert jnp.all(var >= 0), "Variance should be non-negative."


def test_estimator_loo_residuals_squared(setup_data):
    """estimator.loo_residuals_squared should match predictor.loo_residuals_squared."""
    X, y = setup_data
    sigma = 1.0

    est = mellon.FunctionEstimator(sigma=sigma, n_landmarks=0)
    est.fit(X, y)

    var_convenience = est.loo_residuals_squared(X, y)
    var_direct = est.predict.loo_residuals_squared(X, y, sigma=sigma)

    assert jnp.allclose(var_convenience, var_direct), "Convenience method should match direct call."


def test_obs_variance_positive(setup_data):
    """obs_variance should return mostly positive values at training points."""
    X, y = setup_data
    sigma = 1.0

    est = mellon.FunctionEstimator(sigma=sigma, n_landmarks=0, obs_variance=True)
    est.fit(X, y)

    var = est.predict.obs_variance(X)
    # GP smoothing may produce tiny negative values at a few points
    assert jnp.all(var > -0.01), "obs_variance has unexpectedly negative values."
    assert jnp.mean(var > 0) > 0.9, "Most obs_variance values should be positive."


def test_obs_variance_correlates_with_true_noise():
    """obs_variance should correlate with true noise in heteroscedastic data."""
    n = 100
    key = jax.random.PRNGKey(123)
    k1, k2 = jax.random.split(key)
    X = jax.random.normal(k1, (n, 1)) * 2

    # Heteroscedastic noise: higher variance for larger |x|
    true_noise_std = 0.1 + 0.5 * jnp.abs(X[:, 0])
    y = jnp.sin(X[:, 0]) + true_noise_std * jax.random.normal(k2, (n,))

    est = mellon.FunctionEstimator(sigma=0.5, n_landmarks=0, obs_variance=True)
    est.fit(X, y)

    var = est.predict.obs_variance(X)

    corr = _spearman_correlation(true_noise_std**2, var)
    assert corr > 0.3, (
        f"obs_variance should correlate with true noise variance, got Spearman={corr}"
    )


def test_obs_variance_error_when_not_computed(setup_data):
    """Calling obs_variance without obs_variance=True should raise."""
    X, y = setup_data
    sigma = 1.0

    est = mellon.FunctionEstimator(sigma=sigma, n_landmarks=0, obs_variance=False)
    est.fit(X, y)

    with pytest.raises(ValueError, match="obs_variance"):
        est.predict.obs_variance(X)


def test_obs_variance_serialization(setup_data, tmp_path):
    """Serialization round-trip should preserve obs_variance."""
    X, y = setup_data
    sigma = 1.0

    est = mellon.FunctionEstimator(sigma=sigma, n_landmarks=0, obs_variance=True)
    est.fit(X, y)

    var_before = est.predict.obs_variance(X)

    # Serialize to file and deserialize
    filepath = str(tmp_path / "predictor.json")
    est.predict.to_json(filepath)
    pred_restored = mellon.Predictor.from_json(filepath)

    var_after = pred_restored.obs_variance(X)
    assert jnp.allclose(var_before, var_after, atol=1e-6), (
        "obs_variance should survive serialization round-trip."
    )


def test_estimator_get_obs_variance(setup_data):
    """estimator.get_obs_variance() should match predictor.obs_variance()."""
    X, y = setup_data
    sigma = 1.0

    est = mellon.FunctionEstimator(sigma=sigma, n_landmarks=0, obs_variance=True)
    est.fit(X, y)

    var_convenience = est.get_obs_variance(X)
    var_direct = est.predict.obs_variance(X)

    assert jnp.allclose(var_convenience, var_direct), (
        "Convenience method should match direct call."
    )


def test_obs_variance_multi_output():
    """obs_variance should work with multi-output y of shape (n, p)."""
    n, d, p = 80, 3, 4
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    X = jax.random.normal(k1, (n, d))
    y = jax.random.normal(k2, (n, p))

    # Full GP
    est = mellon.FunctionEstimator(sigma=1.0, n_landmarks=0, obs_variance=True)
    est.fit(X, y)
    var = est.predict.obs_variance(X)
    assert var.shape == (n, p), f"Full GP: expected ({n}, {p}), got {var.shape}"

    # Sparse GP
    est_s = mellon.FunctionEstimator(sigma=1.0, n_landmarks=20, obs_variance=True)
    est_s.fit(X, y)
    var_s = est_s.predict.obs_variance(X)
    assert var_s.shape == (n, p), f"Sparse GP: expected ({n}, {p}), got {var_s.shape}"


def test_fit_obs_variance_override(setup_data):
    """fit(x, y, obs_variance=True) should override constructor default."""
    X, y = setup_data
    sigma = 1.0

    est = mellon.FunctionEstimator(sigma=sigma, n_landmarks=0, obs_variance=False)
    est.fit(X, y, obs_variance=True)

    # Should work because we overrode at fit time
    var = est.predict.obs_variance(X)
    assert var.shape == (X.shape[0],)
