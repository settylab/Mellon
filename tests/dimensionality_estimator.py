import pytest
import mellon
import jax
import jax.numpy as jnp


@pytest.fixture
def common_setup_dim_estimator():
    n = 100
    d = 2
    seed = 535
    key = jax.random.PRNGKey(seed)
    L = jax.random.uniform(key, (d, d))
    cov = L.T.dot(L)
    X = jax.random.multivariate_normal(key, jnp.ones(d), cov, (n,))

    est = mellon.DimensionalityEstimator()
    local_dim = est.fit_predict(X)

    dim_std = jnp.std(local_dim)

    def relative_err(dim):
        diff_dim = jnp.std(local_dim - dim) / dim_std
        return diff_dim

    return X, local_dim, relative_err, est, dim_std


def test_dimensionality_estimator_properties(common_setup_dim_estimator):
    X, local_dim, relative_err, est, _ = common_setup_dim_estimator
    n, d = X.shape

    pred = est.predict(X)
    assert (
        relative_err(pred) < 1e-4
    ), "The predictive function should be consistent with the training samples."

    assert len(str(est)) > 0, "The model should have a string representation."

    grads = est.predict_density.gradient(X)
    assert (
        grads.shape == X.shape
    ), "The gradient should have the same shape as the input."

    log_dens = est.predict_density(X)
    assert log_dens.shape == (n,), "There should be one density value per sample."
    hess = est.predict_density.hessian(X)
    assert hess.shape == (n, d, d), "The hessian should have the correct shape."

    result = est.predict_density.hessian_log_determinant(X)
    assert (
        len(result) == 2
    ), "hessian_log_determinant should return signs and lg-values."
    sng, ld = result
    assert sng.shape == (n,), "There should be one sign for each sample."
    assert ld.shape == (n,), "There should be one value for each sample."


def test_dimensionality_estimator_optimizer(common_setup_dim_estimator):
    X, local_dim, relative_err, _, _ = common_setup_dim_estimator

    adam_est = mellon.DimensionalityEstimator(optimizer="adam")
    adam_dim = adam_est.fit_predict(X)
    assert (
        relative_err(adam_dim) < 2e0
    ), "The adam optimizer should produce similar results to the default."


@pytest.mark.parametrize(
    "rank, method, n_landmarks, err_limit",
    [
        (1.0, "percent", 100, 1e0),
        (1.0, "percent", 10, 1e0),
        (0.99, "percent", 80, 1e0),
        (50, "auto", 80, 1e0),
    ],
)
def test_dimensionality_estimator_approximations(
    common_setup_dim_estimator, rank, method, n_landmarks, err_limit
):
    X, local_dim, relative_err, _, _ = common_setup_dim_estimator

    est = mellon.DimensionalityEstimator(
        rank=rank, method=method, n_landmarks=n_landmarks
    )
    est.fit(X)
    dim_appr = est.predict(X)
    assert (
        relative_err(dim_appr) < err_limit
    ), "The approximation should be close to the default."
