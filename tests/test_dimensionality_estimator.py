import pytest
import mellon
import jax
import jax.numpy as jnp


@pytest.fixture
def common_setup_dim_estimator(tmp_path):
    n = 100
    d = 2
    seed = 535
    test_file = tmp_path / "predictor.json"
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

    return X, test_file, local_dim, relative_err, est, dim_std


def test_dimensionality_estimator_properties(common_setup_dim_estimator):
    X, _, local_dim, relative_err, est, _ = common_setup_dim_estimator
    n, d = X.shape

    html_output = est._repr_html_()
    str_output = str(est)

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


@pytest.mark.parametrize(
    "rank, n_landmarks, compress",
    [
        (1.0, 0, None),
        (0.99, 0, None),
        (1.0, 10, None),
        (0.99, 80, None),
    ],
)
def test_dimensionality_estimator_serialization_with_uncertainty(
    common_setup_dim_estimator, rank, n_landmarks, compress
):
    X, test_file, _, _, _, _ = common_setup_dim_estimator
    n = X.shape[0]

    est = mellon.DimensionalityEstimator(
        rank=rank,
        n_landmarks=n_landmarks,
        optimizer="advi",
        predictor_with_uncertainty=True,
    )
    est.fit(X)
    dens_appr = est.predict(X)
    log_dens_appr = est.predict(X, logscale=True)
    is_close = jnp.all(jnp.isclose(dens_appr, jnp.exp(log_dens_appr)))
    assert (
        is_close
    ), "The exp of the log scale prediction should mix the original prediction."
    covariance = est.predict.covariance(X)
    assert covariance.shape == (
        n,
    ), "The diagonal of the covariance matrix should be repoorted."
    mean_covariance = est.predict.mean_covariance(X)
    assert mean_covariance.shape == (
        n,
    ), "The diagonal of the mean covariance should be repoorted."
    uncertainty_pred = est.predict.uncertainty(X)
    assert uncertainty_pred.shape == (n,), "One value per sample should be reported."

    # Test serialization
    est.predict.to_json(test_file, compress=compress)
    predictor = mellon.Predictor.from_json(test_file, compress=compress)
    reprod = predictor(X)
    is_close = jnp.all(jnp.isclose(dens_appr, reprod))
    assert_msg = "Serialized + deserialized predictor should produce the same results."
    assert is_close, assert_msg
    reprod_uncertainty = predictor.uncertainty(X)
    is_close = jnp.all(jnp.isclose(uncertainty_pred, reprod_uncertainty))
    assert_msg = "Serialized + deserialized predictor should produce the same uncertainty results."
    assert is_close, assert_msg


def test_dimensionality_estimator_optimizer(common_setup_dim_estimator):
    X, _, local_dim, relative_err, _, _ = common_setup_dim_estimator

    adam_est = mellon.DimensionalityEstimator(optimizer="adam")
    adam_dim = adam_est.fit_predict(X)
    assert (
        relative_err(adam_dim) < 2e0
    ), "The adam optimizer should produce similar results to the default."


@pytest.mark.parametrize(
    "rank, n_landmarks, err_limit",
    [
        (1.0, 100, 1e0),
        (1.0, 10, 2e0),
        (0.99, 80, 1e0),
        (50, 80, 1e0),
    ],
)
def test_dimensionality_estimator_approximations(
    common_setup_dim_estimator, rank, n_landmarks, err_limit
):
    X, _, local_dim, relative_err, _, _ = common_setup_dim_estimator

    est = mellon.DimensionalityEstimator(rank=rank, n_landmarks=n_landmarks)
    est.fit(X)
    dim_appr = est.predict(X)
    assert (
        relative_err(dim_appr) < err_limit
    ), "The approximation should be close to the default."


def test_dimensionality_estimator_errors(common_setup_dim_estimator):
    X, _, _, _, _, _ = common_setup_dim_estimator
    lX = jnp.concatenate(
        [
            X,
        ]
        * 26,
        axis=1,
    )
    est = mellon.DimensionalityEstimator()

    with pytest.raises(ValueError):
        est.fit_predict()
    with pytest.raises(ValueError):
        est.fit(None)
    est.set_x(X)
    with pytest.raises(ValueError):
        est.prepare_inference(lX)
    loss_func, initial_value = est.prepare_inference(None)
    est.run_inference(loss_func, initial_value, "advi")
    est.process_inference(est.pre_transformation)
    with pytest.raises(ValueError):
        est.fit_predict(lX)
    est.fit_predict()
