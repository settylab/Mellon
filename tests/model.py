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
    assert log_dens.shape == (n,), \
        "There should be one density value for each sample."
    d_std = jnp.std(log_dens)
    def relative_err(dens):
        diff = jnp.abs(log_dens - dens)
        return jnp.max(diff) / d_std
    pred_log_dens = est.predict(X)
    assert relative_err(pred_log_dens) < 1e-5, \
        "The predicive function should be consistent with the denisty on " \
        "the training samples."

    est_full = mellon.DensityEstimator(rank=1., method="percent", n_landmarks=n)
    full_log_dens = est_full.fit_predict(X)
    assert relative_err(full_log_dens) < 1e-1, \
        "The default approximation should be close to the full rank result."

    est = mellon.DensityEstimator(rank=1., method="percent", n_landmarks=10)
    dens_appr = est.fit_predict(X)
    assert relative_err(dens_appr) < 2e-1, \
        "The low landmarks approximation should be close to the default."

    est = mellon.DensityEstimator(rank=.99, method="percent", n_landmarks=80)
    dens_appr = est.fit_predict(X)
    assert relative_err(dens_appr) < 2e-1, \
        "The low landmarks + Nystrom approximation should be close to the default."

    est = mellon.DensityEstimator(rank=50, n_landmarks=80)
    dens_appr = est.fit_predict(X)
    assert relative_err(dens_appr) < 2e-1, \
        "The low landmarks + Nystrom approximation should be close to the default."
