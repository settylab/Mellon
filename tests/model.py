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
        "The predicive function should be consistent with the denisty on "
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

    est_full = mellon.DensityEstimator(rank=1.0, method="percent", n_landmarks=n)
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
    y = jnp.sum(jnp.sin(X / 2), axis=1)

    est = mellon.FunctionEstimator()
    pred = est.fit_predict(X, y)
    assert pred.shape == (n,), "There should be a predicted value for each sample."

    assert len(str(est)) > 0, "The model should have a string representation."

    err = jnp.std(y - pred)
    assert err < 1e-5, "The prediction should be close to the intput value."

    est_full = mellon.FunctionEstimator(rank=1.0, method="percent", n_landmarks=0)
    full_pred = est_full.fit_predict(X, y)
    err = jnp.max(jnp.abs(full_pred - pred))
    assert (
        err < 1e-4
    ), "The default approximation should be close to the full rank result."

    est = mellon.FunctionEstimator(rank=1.0, method="percent", n_landmarks=10)
    pred_appr = est.fit_predict(X, y)
    err = jnp.std(pred_appr - pred)
    assert err < 1e-1, "The low landmarks approximation should be close to the default."

    est = mellon.FunctionEstimator(rank=0.99, method="percent", n_landmarks=80)
    pred_appr = est.fit_predict(X, y)
    err = jnp.std(pred_appr - pred)
    assert (
        err < 1e-1
    ), "The low landmarks + Nystrom approximation should be close to the default."

    est = mellon.FunctionEstimator(rank=50, n_landmarks=80)
    pred_appr = est.fit_predict(X, y)
    err = jnp.std(pred_appr - pred)
    assert (
        err < 1e-1
    ), "The low landmarks + Nystrom approximation should be close to the default."

    est = mellon.FunctionEstimator()
    d1_pred = est.fit_predict(X[:, 0], y)
    assert d1_pred.shape == (n,), "There should be one result per sample."

    est = mellon.FunctionEstimator(rank=1.0, method="percent", n_landmarks=0)
    d1_pred_full = est.fit_predict(X[:, 0], y)
    assert (
        jnp.std(d1_pred - d1_pred_full) < 1e-5
    ), "The scalar state function estimations be consistent under approximation."
