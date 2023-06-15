import pytest
import mellon
import jax
import jax.numpy as jnp


@pytest.fixture
def common_setup(tmp_path):
    n = 100
    d = 2
    seed = 535
    test_file = tmp_path / "predictor.json"
    logger = mellon.Log()
    key = jax.random.PRNGKey(seed)
    L = jax.random.uniform(key, (d, d))
    cov = L.T.dot(L)
    X = jax.random.multivariate_normal(key, jnp.ones(d), cov, (n,))

    est = mellon.DensityEstimator()
    log_dens = est.fit_predict(X)
    d_std = jnp.std(log_dens)

    def relative_err(dens):
        diff = jnp.std(log_dens - dens)
        return diff / d_std

    return X, test_file, logger, relative_err, est, d_std


def test_density_estimator_properties(common_setup):
    X, _, _, relative_err, est, _ = common_setup
    n, d = X.shape

    pred_log_dens = est.predict(X)
    assert relative_err(pred_log_dens) < 1e-5, (
        "The predicive function should be consistent with the density on "
        "the training samples."
    )

    grads = est.predict.gradient(X)
    assert (
        grads.shape == X.shape
    ), "The gradient should have the same shape as the input."

    hess = est.predict.hessian(X)
    assert hess.shape == (n, d, d), "The hessian should have the correct shape."

    result = est.predict.hessian_log_determinant(X)
    assert (
        len(result) == 2
    ), "hessian_log_determinan should return signes and lg-values."
    sng, ld = result
    assert sng.shape == (n,), "There should be one sign for each hessian determinan."
    assert ld.shape == (n,), "There should be one value for each hessian determinan."

    assert len(str(est)) > 0, "The model should have a string representation."


def test_density_estimator_fractal_dimension(common_setup):
    X, _, _, _, _, _ = common_setup
    n = X.shape[0]

    frac_est = mellon.DensityEstimator(d_method="fractal")
    log_dens_frac = frac_est.fit_predict(X)
    assert (
        frac_est.d != X.shape[1]
    ), "The fractal dimension should not equal the embedding dimension"
    assert log_dens_frac.shape == (
        n,
    ), "There should be one density value for each sample."


def test_density_estimator_optimizers(common_setup):
    X, _, _, relative_err, _, _ = common_setup

    adam_est = mellon.DensityEstimator(optimizer="adam")
    adam_dens = adam_est.fit_predict(X)
    assert (
        relative_err(adam_dens) < 1e-3
    ), "The adam optimizer should produce similar results to the default."


@pytest.mark.parametrize(
    "rank, method, n_landmarks, err_limit",
    [
        (1.0, "percent", 0, 1e-1),
        (1.0, "percent", 10, 2e-1),
        (0.99, "percent", 80, 2e-1),
    ],
)
def test_density_estimator_approximations(
    common_setup, rank, method, n_landmarks, err_limit
):
    X, _, _, relative_err, _, _ = common_setup

    est = mellon.DensityEstimator(rank=rank, method=method, n_landmarks=n_landmarks)
    est.fit(X)
    dens_appr = est.predict(X)
    assert (
        relative_err(dens_appr) < err_limit
    ), "The approximation should be close to the default."


@pytest.mark.parametrize(
    "rank, n_landmarks, compress",
    [
        (1.0, 0, None),
        (1.0, 10, None),
        (0.99, 80, None),
        (0.99, 80, "gzip"),
        (0.99, 80, "bz2"),
    ],
)
def test_density_estimator_serialization(common_setup, rank, n_landmarks, compress):
    X, test_file, logger, _, _, _ = common_setup

    est = mellon.DensityEstimator(rank=rank, n_landmarks=n_landmarks)
    est.fit(X)
    dens_appr = est.predict(X)

    # Test serialization
    est.predict.to_json(test_file, compress=compress)
    logger.info(f"Serialized the predictor and saved it to {test_file}.")
    predictor = mellon.Predictor.from_json(test_file, compress=compress)
    logger.info("Deserialized the predictor from the JSON file.")
    reprod = predictor(X)
    logger.info("Made a prediction with the deserialized predictor.")
    is_close = jnp.all(jnp.isclose(dens_appr, reprod))
    assert_msg = "Serialized + deserialized predictor should produce the same results."
    assert is_close, assert_msg
    logger.info(
        "Assertion passed: the deserialized predictor produced the expected results."
    )


def test_density_estimator_dictionary_serialization(common_setup):
    X, _, logger, _, est, _ = common_setup
    dens_appr = est.predict(X)

    # Test dictionay serialization
    data_dict = est.predict.to_dict()
    assert isinstance(data_dict, dict), "Predictor.to_dict() must return a dictionary."
    logger.info("Serialized the predictor to dictionary.")
    predictor = mellon.Predictor.from_dict(data_dict)
    logger.info("Deserialized the predictor from the dictionary.")
    reprod = predictor(X)
    logger.info("Made a prediction with the deserialized predictor.")
    is_close = jnp.all(jnp.isclose(dens_appr, reprod))
    assert_msg = "Serialized + deserialized predictor should produce the same results."
    assert is_close, assert_msg
    logger.info(
        "Assertion passed: the deserialized predictor produced the expected results."
    )


def test_density_estimator_single_dimension(common_setup):
    X, _, _, _, _, _ = common_setup
    n = X.shape[0]

    est = mellon.DensityEstimator()
    d1_pred = est.fit_predict(X[:, 0])
    assert d1_pred.shape == (n,), "There should be one result per sample."

    est = mellon.DensityEstimator(rank=1.0, method="percent", n_landmarks=0)
    d1_pred_full = est.fit_predict(X[:, 0])
    assert (
        jnp.std(d1_pred - d1_pred_full) < 1e-2
    ), "The scalar state function estimations be consistent under approximation."
