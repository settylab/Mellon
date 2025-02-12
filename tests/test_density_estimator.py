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
    logger = mellon.logger
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

    html_output = est._repr_html_()
    str_output = str(est)

    len_str = len(str(mellon.DensityEstimator()))
    assert len_str > 0, "The model should have a string representation."

    pred_log_dens = est.predict(X)
    assert relative_err(pred_log_dens) < 1e-5, (
        "The predicive function should be consistent with the density on "
        "the training samples."
    )

    pred_str = len(str(est.predict))
    assert pred_str > 0, "The predictor should have a string representation."

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

    assert (
        len(str(est)) > len_str
    ), "The model should have a longer string representation after fitting."


def test_density_estimator_optimizers(common_setup):
    X, _, _, relative_err, _, _ = common_setup

    adam_est = mellon.DensityEstimator(optimizer="adam")
    adam_dens = adam_est.fit_predict(X)
    assert (
        relative_err(adam_dens) < 2e-3
    ), "The adam optimizer should produce similar results to the default."


@pytest.mark.parametrize(
    "rank, n_landmarks, err_limit",
    [
        (1.0, 0, 1e-1),
        (1.0, 10, 2e-1),
        (0.99, 80, 2e-1),
    ],
)
def test_density_estimator_approximations(common_setup, rank, n_landmarks, err_limit):
    X, _, _, relative_err, _, _ = common_setup

    est = mellon.DensityEstimator(rank=rank, n_landmarks=n_landmarks)
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
    norm_dens_appr = est.predict(X, normalize=True)
    is_close = jnp.all(jnp.isclose(dens_appr, norm_dens_appr))
    assert not is_close, "The normalized and non-normalized predictions should differ."

    # Test serialization
    json_string = est.predict.to_json()
    assert isinstance(
        json_string, str
    ), "Json string should be returned if no filename is given."
    est.predict.to_json(test_file, compress=compress)
    est.predict.to_json(str(test_file), compress=compress)
    logger.info(f"Serialized the predictor and saved it to {test_file}.")
    predictor = mellon.Predictor.from_json(test_file, compress=compress)
    logger.info("Deserialized the predictor from the JSON file.")
    reprod = predictor(X)
    norm_reprod = predictor(X, normalize=True)
    logger.info("Made a prediction with the deserialized predictor.")
    is_close = jnp.all(jnp.isclose(dens_appr, reprod))
    norm_is_close = jnp.all(jnp.isclose(norm_dens_appr, norm_reprod))
    assert_msg = "Serialized + deserialized predictor should produce the same results."
    assert is_close and norm_is_close, assert_msg
    logger.info(
        "Assertion passed: the deserialized predictor produced the expected results."
    )
    logger.info("Serializing deserialized predictor again.")
    predictor.to_json(test_file, compress=compress)
    # test backwards compatibility
    edict = predictor.to_dict()
    edict["metadata"]["module_version"] = "1.3.1"
    edict["data"].pop("n_obs")
    edict["data"].pop("_state_variables")
    mellon.Predictor.from_dict(edict)
    reprod = predictor(X, normalize=False)
    is_close = jnp.all(jnp.isclose(dens_appr, reprod))
    assert (
        is_close
    ), "Deserialized predictor of mellon 1.3.1 should produce the same results."


def test_density_estimator_without_uncertainty(common_setup):
    X, _, _, _, est, _ = common_setup

    with pytest.raises(ValueError):
        est.predict.covariance(X)
    with pytest.raises(ValueError):
        est.predict.mean_covariance(X)
    with pytest.raises(ValueError):
        est.predict.uncertainty(X)


@pytest.mark.parametrize(
    "rank, n_landmarks, compress",
    [
        (1.0, 0, None),
        (0.99, 0, None),
        (1.0, 10, None),
        (0.99, 80, None),
    ],
)
def test_density_estimator_serialization_with_uncertainty(
    common_setup, rank, n_landmarks, compress
):
    X, test_file, logger, _, _, _ = common_setup
    n = X.shape[0]

    est = mellon.DensityEstimator(
        rank=rank,
        n_landmarks=n_landmarks,
        optimizer="advi",
        predictor_with_uncertainty=True,
    )
    est.fit(X)
    dens_appr = est.predict(X)
    covariance = est.predict.covariance(X)
    assert covariance.shape == (
        n,
    ), "The diagonal of the covariance matrix should be reported."
    mean_covariance = est.predict.mean_covariance(X)
    assert mean_covariance.shape == (
        n,
    ), "The diagonal of the mean covariance should be reported."
    uncertainty_pred = est.predict.uncertainty(X)
    assert uncertainty_pred.shape == (n,), "One value per sample should be reported."

    full_covariance = est.predict.covariance(X, diag=False)
    assert full_covariance.shape == (
        n,
        n,
    ), "The full covariance matrix should be repoorted."
    full_mean_covariance = est.predict.mean_covariance(X, diag=False)
    assert full_mean_covariance.shape == (
        n,
        n,
    ), "The full mean covariance should be repoorted."
    full_uncertainty_pred = est.predict.uncertainty(X, diag=False)
    assert full_uncertainty_pred.shape == (
        n,
        n,
    ), "The full covariance should be reported."

    # Test serialization
    est.predict.to_json(test_file, compress=compress)
    logger.info(
        f"Serialized the predictor with uncertainty and saved it to {test_file}."
    )
    predictor = mellon.Predictor.from_json(test_file, compress=compress)
    logger.info("Deserialized the predictor with uncertainty from the JSON file.")
    reprod = predictor(X)
    logger.info("Made a prediction with the deserialized predictor.")
    is_close = jnp.all(jnp.isclose(dens_appr, reprod))
    assert_msg = "Serialized + deserialized predictor should produce the same results."
    assert is_close, assert_msg
    reprod_uncertainty = predictor.uncertainty(X)
    logger.info("Made a uncertainty prediction with the deserialized predictor.")
    is_close = jnp.all(jnp.isclose(uncertainty_pred, reprod_uncertainty))
    assert_msg = "Serialized + deserialized predictor should produce the same uncertainty results."
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

    est = mellon.DensityEstimator(rank=1.0, n_landmarks=0)
    d1_pred_full = est.fit_predict(X[:, 0])
    assert (
        jnp.std(d1_pred - d1_pred_full) < 1e-2
    ), "The scalar state function estimations be consistent under approximation."


def test_density_estimator_errors(common_setup):
    X, test_file, _, _, _, _ = common_setup
    lX = jnp.concatenate(
        [
            X,
        ]
        * 2,
        axis=1,
    )
    est = mellon.DensityEstimator()

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
    predictor = est.predict
    with pytest.raises(ValueError):
        predictor(X[:, :-1])
    with pytest.raises(ValueError):
        predictor.covariance(X[:, :-1])
    with pytest.raises(ValueError):
        predictor.mean_covariance(X[:, :-1])
    with pytest.raises(ValueError):
        predictor.uncertainty(X[:, :-1])
    with pytest.raises(ValueError):
        predictor.to_json(test_file, compress="bad_type")
    est.fit_predict()
    est.predict.n_obs = None
    with pytest.raises(ValueError):
        est.predict(X, normalize=True)
    est = mellon.DensityEstimator(predictor_with_uncertainty=True)
    with pytest.raises(ValueError):
        est.fit(X)
