import pytest
import mellon
import jax
import jax.numpy as jnp


@pytest.fixture
def common_setup_time_sensitive(tmp_path):
    n_per_batch = 10
    n_batches = 4
    test_time = 2
    n = n_per_batch * n_batches
    d = 2
    seed = 535
    test_file = tmp_path / "predictor.json"
    logger = mellon.logger
    key = jax.random.PRNGKey(seed)
    L = jax.random.uniform(key, (d, d))
    cov = L.T.dot(L)
    X = jax.random.multivariate_normal(key, jnp.ones(d), cov, (n,))
    times = jnp.repeat(jnp.arange(n_batches), n_per_batch)

    est = mellon.TimeSensitiveDensityEstimator()
    log_dens = est.fit_predict(X, times)
    d_std = jnp.std(log_dens)

    def relative_err(dens):
        diff = jnp.std(log_dens - dens)
        return diff / d_std

    return X, times, test_file, logger, relative_err, est, d_std, test_time


def test_time_sensitive_density_estimator_properties(common_setup_time_sensitive):
    X, times, _, _, relative_err, est, _, test_time = common_setup_time_sensitive
    n, d = X.shape
    multi_time = [test_time, test_time, test_time + 1]
    n_times = len(multi_time)

    html_output = est._repr_html_()
    str_output = str(est)

    pred_log_dens = est.predict(X, times)
    assert relative_err(pred_log_dens) < 1e-5, (
        "The predicive function should be consistent with the density on "
        "the training samples."
    )

    grads = est.predict.gradient(X, test_time)
    assert (
        grads.shape == X.shape
    ), "The gradient should have the same shape as the input."

    hess = est.predict.hessian(X, test_time)
    assert hess.shape == (n, d, d), "The hessian should have the correct shape."
    hess = est.predict.hessian(X, multi_time=multi_time)
    assert hess.shape == (
        n,
        n_times,
        d,
        d,
    ), "The hessians should have the correct shape."
    assert jnp.all(
        hess[:, 0, :, :] == hess[:, 1, :, :]
    ), "Equal time points should produce equal results."
    assert jnp.any(
        hess[:, 0, :, :] != hess[:, 2, :, :]
    ), "Different time points should produce differnt results."

    result = est.predict.hessian_log_determinant(X, test_time)
    assert (
        len(result) == 2
    ), "hessian_log_determinan should return signes and lg-values."
    sng, ld = result
    assert sng.shape == (n,), "There should be one sign for each hessian determinan."
    assert ld.shape == (n,), "There should be one value for each hessian determinan."

    time_d = est.predict.time_derivative(X, test_time)
    assert time_d.shape == (n,), "The time derivative should have one value per sample."

    assert len(str(est)) > 0, "The model should have a string representation."


@pytest.mark.parametrize(
    "rank, n_landmarks, err_limit",
    [
        (1.0, 10, 2e-1),
        (0.99, 80, 5e-1),
    ],
)
def test_time_sensitive_density_estimator_approximations(
    common_setup_time_sensitive, rank, n_landmarks, err_limit
):
    X, times, _, _, relative_err, _, _, _ = common_setup_time_sensitive
    n = X.shape[0]

    est = mellon.TimeSensitiveDensityEstimator(
        rank=rank,
        n_landmarks=n_landmarks,
        _save_intermediate_ls_times=True,
        normalize_per_time_point=True,
    )
    est.fit(X, times)
    dens_appr = est.predict(X, times)
    if n_landmarks < n:
        assert (
            est.landmarks is not None
        ), f"Since n_landmarks ({n_landmarks}) < n ({n}) landmarks should be computed."
        assert (
            est.landmarks.shape[0] == n_landmarks
        ), "The right number of landmarks was not produced."
    assert (
        relative_err(dens_appr) < err_limit
    ), "The approximation should be close to the default."
    assert hasattr(
        est, "densities"
    ), "est.densities should be stored since _save_intermediate_ls_times=True was passed."
    assert hasattr(
        est, "predictors"
    ), "est.predictors should be stored since _save_intermediate_ls_times=True was passed."
    assert hasattr(
        est, "numeric_stages"
    ), "est.numeric_stages should be stored since _save_intermediate_ls_times=True was passed."


@pytest.mark.parametrize(
    "rank, n_landmarks, compress",
    [
        (1.0, 10, None),
        (0.8, 10, None),
        (0.99, 80, "gzip"),
        (0.99, 80, "bz2"),
    ],
)
def test_time_sensitive_density_estimator_serialization(
    common_setup_time_sensitive, rank, n_landmarks, compress
):
    X, times, test_file, logger, _, _, _, _ = common_setup_time_sensitive

    est = mellon.TimeSensitiveDensityEstimator(rank=rank, n_landmarks=n_landmarks)
    est.fit(X, times)
    dens_appr = est.predict(X, times)

    # Test serialization
    est.predict.to_json(test_file, compress=compress)
    logger.info(f"Serialized the predictor and saved it to {test_file}.")
    predictor = mellon.Predictor.from_json(test_file, compress=compress)
    logger.info("Deserialized the predictor from the JSON file.")
    reprod = predictor(X, times)
    logger.info("Made a prediction with the deserialized predictor.")
    is_close = jnp.all(jnp.isclose(dens_appr, reprod))
    assert_msg = "Serialized + deserialized predictor should produce the same results."
    assert is_close, assert_msg
    logger.info(
        "Assertion passed: the deserialized predictor produced the expected results."
    )


@pytest.mark.parametrize(
    "rank, n_landmarks, compress",
    [
        (1.0, 10, None),
        (0.8, 10, None),
        (0.99, 80, "gzip"),
        (0.99, 80, "bz2"),
    ],
)
def test_density_estimator_serialization_with_uncertainty(
    common_setup_time_sensitive, rank, n_landmarks, compress
):
    X, times, test_file, logger, _, _, _, _ = common_setup_time_sensitive
    n = X.shape[0]

    est = mellon.TimeSensitiveDensityEstimator(
        rank=rank,
        n_landmarks=n_landmarks,
        optimizer="advi",
        predictor_with_uncertainty=True,
    )
    est.fit(X, times)
    dens_appr = est.predict(X, times)
    covariance = est.predict.covariance(X, times)
    assert covariance.shape == (
        n,
    ), "The diagonal of the covariance matrix should be repoorted."
    mean_covariance = est.predict.mean_covariance(X, times)
    assert mean_covariance.shape == (
        n,
    ), "The diagonal of the mean covariance should be repoorted."
    uncertainty_pred = est.predict.uncertainty(X, times)
    assert uncertainty_pred.shape == (n,), "One value per sample should be reported."

    # Test serialization
    est.predict.to_json(test_file, compress=compress)
    logger.info(
        f"Serialized the predictor with uncertainty and saved it to {test_file}."
    )
    predictor = mellon.Predictor.from_json(test_file, compress=compress)
    logger.info("Deserialized the predictor with uncertainty from the JSON file.")
    reprod = predictor(X, times)
    logger.info("Made a prediction with the deserialized predictor.")
    is_close = jnp.all(jnp.isclose(dens_appr, reprod))
    assert_msg = "Serialized + deserialized predictor should produce the same results."
    assert is_close, assert_msg
    reprod_uncertainty = predictor.uncertainty(X, times)
    logger.info("Made a uncertainty prediction with the deserialized predictor.")
    is_close = jnp.all(jnp.isclose(uncertainty_pred, reprod_uncertainty))
    assert_msg = "Serialized + deserialized predictor should produce the same uncertainty results."
    assert is_close, assert_msg
    logger.info(
        "Assertion passed: the deserialized predictor produced the expected results."
    )


def test_density_estimator_errors(common_setup_time_sensitive):
    X, times, _, _, _, _, _, _ = common_setup_time_sensitive
    Xt = jnp.concatenate([X, times[:, None]], axis=1)
    lX = jnp.concatenate(
        [
            X,
        ]
        * 26
        + [
            times[:, None],
        ],
        axis=1,
    )
    est = mellon.TimeSensitiveDensityEstimator()

    with pytest.raises(ValueError):
        est.fit_predict()
    with pytest.raises(ValueError):
        est.fit(None)
    est.set_x(Xt)
    with pytest.raises(ValueError):
        est.prepare_inference(lX)
    loss_func, initial_value = est.prepare_inference(None)
    est.run_inference(loss_func, initial_value, "advi")
    est.process_inference(est.pre_transformation)
    with pytest.raises(ValueError):
        est.predict(X[:, :-1], times)
    with pytest.raises(ValueError):
        est.fit_predict(lX)
    est.fit_predict()
    est.predict.n_obs = None
    with pytest.raises(ValueError):
        est.predict(X, time=times, normalize=True)


@pytest.mark.parametrize(
    "normalization, different",
    [
        (False, False),
        (True, False),
        ([4, 4, 1000, 4], True),
        (jnp.array([4, 4, 1000, 4]), True),
        ({1: 4, 0: 4, 2: 1000, 3: 4}, True),
    ],
)
def test_time_sensitive_density_estimator_normalizations(
    common_setup_time_sensitive, normalization, different
):
    X, times, _, _, relative_err, _, _, _ = common_setup_time_sensitive
    err_limit = 1e-4
    min_diff = 1e-1

    est = mellon.TimeSensitiveDensityEstimator(
        normalize_per_time_point=normalization,
    )
    est.fit(X, times)
    dens_appr = est.predict(X, times)
    if different:
        assert (
            relative_err(dens_appr) > min_diff
        ), "This normalization should be different the default."
    else:
        assert (
            relative_err(dens_appr) < err_limit
        ), "This normalization should be close to the default."
