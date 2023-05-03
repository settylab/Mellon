import mellon
import jax
import jax.numpy as jnp


def test_DensityEstimator():
    n = 100
    d = 2
    seed = 535
    key = jax.random.PRNGKey(seed)
    L = jax.random.uniform(key, (d, d))
    cov = L.T.dot(L)
    X = jax.random.multivariate_normal(key, jnp.ones(d), cov, (n,))

    est = mellon.DensityEstimator()
    log_dens = est.fit_predict(X)
    assert log_dens.shape == (n,), "There should be one density value for each sample."
    d_std = jnp.std(log_dens)

    def relative_err(dens):
        diff = jnp.std(log_dens - dens)
        return diff / d_std

    pred_log_dens = est.predict(X)
    assert relative_err(pred_log_dens) < 1e-5, (
        "The predicive function should be consistent with the density on "
        "the training samples."
    )

    grads = est.gradient(X)
    assert (
        grads.shape == X.shape
    ), "The gradient should have the same shape as the input."

    hess = est.hessian(X)
    assert hess.shape == (n, d, d), "The hessian should have the correct shape."

    result = est.hessian_log_determinant(X)
    assert (
        len(result) == 2
    ), "hessian_log_determinan should return signes and lg-values."
    sng, ld = result
    assert sng.shape == (n,), "There should be one sign for each hessian determinan."
    assert ld.shape == (n,), "There should be one value for each hessian determinan."

    assert len(str(est)) > 0, "The model should have a string representation."

    frac_est = mellon.DensityEstimator(d_method="fractal")
    log_dens_frac = frac_est.fit_predict(X)
    assert (
        frac_est.d != X.shape[1]
    ), "The fractal dimension should not equal the embedding dimension"
    assert log_dens_frac.shape == (
        n,
    ), "There should be one density value for each sample."

    adam_est = mellon.DensityEstimator(optimizer="adam")
    adam_dens = adam_est.fit_predict(X)
    assert (
        relative_err(adam_dens) < 1e-3
    ), "The adam optimizer should produce similar results to the default."

    est_full = mellon.DensityEstimator(rank=1.0, method="percent", n_landmarks=0)
    est_full.fit(X)
    full_log_dens = est_full.predict(X)
    assert (
        relative_err(full_log_dens) < 1e-1
    ), "The default approximation should be close to the full rank result."

    est = mellon.DensityEstimator(rank=1.0, method="percent", n_landmarks=10)
    est.fit(X)
    dens_appr = est.predict(X)
    assert (
        relative_err(dens_appr) < 2e-1
    ), "The low landmarks approximation should be close to the default."

    est = mellon.DensityEstimator(rank=0.99, method="percent", n_landmarks=80)
    est.fit(X)
    dens_appr = est.predict(X)
    assert (
        relative_err(dens_appr) < 2e-1
    ), "The low landmarks + Nystrom approximation should be close to the default."

    est = mellon.DensityEstimator(rank=50, n_landmarks=80)
    est.fit(X)
    dens_appr = est.predict(X)
    assert (
        relative_err(dens_appr) < 2e-1
    ), "The low landmarks + Nystrom approximation should be close to the default."

    est = mellon.DensityEstimator()
    d1_pred = est.fit_predict(X[:, 0])
    assert d1_pred.shape == (n,), "There should be one result per sample."

    est = mellon.DensityEstimator(rank=1.0, method="percent", n_landmarks=0)
    d1_pred_full = est.fit_predict(X[:, 0])
    assert (
        jnp.std(d1_pred - d1_pred_full) < 1e-2
    ), "The scalar state function estimations be consistent under approximation."


def test_FunctionEstimator():
    n = 100
    d = 2
    seed = 535
    key = jax.random.PRNGKey(seed)
    L = jax.random.uniform(key, (d, d))
    cov = L.T.dot(L)
    X = jax.random.multivariate_normal(key, jnp.ones(d), cov, (n,))
    noise = 1e-4 * jnp.sum(jnp.sin(X * 1e16), axis=1)
    noiseless_y = jnp.sum(jnp.sin(X / 2), axis=1)
    y = noiseless_y + noise
    Y = jnp.stack([y, y - noise])

    est = mellon.FunctionEstimator(sigma=1e-3)
    pred = est.fit_predict(X, y)
    assert pred.shape == (n,), "There should be a predicted value for each sample."

    assert len(str(est)) > 0, "The model should have a string representation."

    err = jnp.std(y - pred)
    assert err < 1e-4, "The prediction should be close to the intput value."

    err = jnp.std(noiseless_y - pred)
    assert err < 1e-4, "The prediction should be close to the true value."

    m_pred = est.multi_fit_predict(X, Y, X)
    assert m_pred.shape == (
        2,
        n,
    ), "There should be a value for each sample and location."

    est_full = mellon.FunctionEstimator(sigma=1e-3, n_landmarks=0)
    full_pred = est_full.fit_predict(X, y)
    err = jnp.max(jnp.abs(full_pred - pred))
    assert (
        err < 1e-4
    ), "The default approximation should be close to the full rank result."

    m_pred_full = est_full.multi_fit_predict(X, Y, X)
    assert (
        jnp.mean(jnp.std(m_pred - m_pred_full)) < 1e-50
    ), "The approximated multipredict should be consistent with the full one."

    est = mellon.FunctionEstimator(sigma=1e-3, n_landmarks=10)
    pred_appr = est.fit_predict(X, y)
    err = jnp.std(pred_appr - pred)
    assert err < 1e-1, "The low landmarks approximation should be close to the default."


    m_pred_xu = est_full.multi_fit_predict(X, Y, est.landmarks)
    m_pred_app = est.multi_fit_predict(X, Y, est.landmarks)
    assert m_pred_xu.shape == (
        2,
        est.landmarks.shape[0],
    ), "There should be a value for each sample and location."
    assert (
        jnp.mean(jnp.std(m_pred_xu - m_pred_app)) < 1e-1
    ), "The approximated multipredict should be consistent with the default one."

    est = mellon.FunctionEstimator(sigma=1e-3)
    d1_pred = est.fit_predict(X[:, 0], y)
    assert d1_pred.shape == (n,), "There should be one result per sample."

    est = mellon.FunctionEstimator(sigma=1e-3, n_landmarks=0)
    d1_pred_full = est.fit_predict(X[:, 0], y)
    assert (
        jnp.std(d1_pred - d1_pred_full) < 1e-5
    ), "The scalar state function estimations be consistent under approximation."

    Y = jnp.stack([y, y])
    m_pred = est.multi_fit_predict(X, Y)
    assert jnp.std(m_pred - pred[None, :]) < 1e-4, (
        "The scalar multi function estimations should be consistent with the "
        "single function estimation."
    )


def test_DimensionalityEstimator():
    n = 100
    d = 2
    seed = 535
    key = jax.random.PRNGKey(seed)
    L = jax.random.uniform(key, (d, d))
    cov = L.T.dot(L)
    X = jax.random.multivariate_normal(key, jnp.ones(d), cov, (n,))

    est = mellon.DimensionalityEstimator()
    local_dim = est.fit_predict(X)
    assert local_dim.shape == (
        n,
    ), "There should be one dimensionality value for each sample."
    dim_std = jnp.std(local_dim)

    def relative_err(dim):
        diff_dim = jnp.std(local_dim - dim) / dim_std
        return diff_dim

    pred = est.predict(X)
    assert (
        relative_err(pred) < 1e-5
    ), "The predicive function should be consistent with the training samples."

    assert len(str(est)) > 0, "The model should have a string representation."

    adam_est = mellon.DimensionalityEstimator(optimizer="adam")
    adam_dim = adam_est.fit_predict(X)
    assert (
        relative_err(adam_dim) < 1e0
    ), "The adam optimizer should produce similar results to the default."

    est_full = mellon.DimensionalityEstimator(rank=1.0, method="percent", n_landmarks=n)
    est_full.fit(X)
    full_local_dim = est_full.predict(X)
    assert (
        relative_err(full_local_dim) < 1e0
    ), "The default approximation should be close to the full rank result."

    est = mellon.DimensionalityEstimator(rank=1.0, method="percent", n_landmarks=10)
    est.fit(X)
    dim_appr = est.predict(X)
    assert (
        relative_err(dim_appr) < 1e0
    ), "The low landmarks approximation should be close to the default."

    est = mellon.DimensionalityEstimator(rank=0.99, method="percent", n_landmarks=80)
    est.fit(X)
    dim_appr = est.predict(X)
    assert (
        relative_err(dim_appr) < 1e0
    ), "The low landmarks + Nystrom approximation should be close to the default."

    est = mellon.DimensionalityEstimator(rank=50, n_landmarks=80)
    est.fit(X)
    dim_appr = est.predict(X)
    assert (
        relative_err(dim_appr) < 1e0
    ), "The low landmarks + Nystrom approximation should be close to the default."

    grads = est.gradient_density(X)
    assert (
        grads.shape == X.shape
    ), "The gradient should have the same shape as the input."

    log_dens = est.predict_density(X)
    hess = est.hessian_density(X)
    assert hess.shape == (n, d, d), "The hessian should have the correct shape."

    result = est.hessian_log_determinant_density(X)
    assert (
        len(result) == 2
    ), "hessian_log_determinan should return signes and lg-values."
    sng, ld = result
    assert sng.shape == (n,), "There should be one sign for each hessian determinan."
    assert ld.shape == (n,), "There should be one value for each hessian determinan."
