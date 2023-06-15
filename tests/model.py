import mellon
import jax
import jax.numpy as jnp


def test_DensityEstimator(tmp_path):
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

    # Test serialization
    est_full.predict.to_json(test_file)
    logger.info(f"Serialized the predictor and saved it to {test_file}.")
    predictor = mellon.Predictor.from_json(test_file)
    logger.info("Deserialized the predictor from the JSON file.")
    reprod = predictor(X)
    logger.info("Made a prediction with the deserialized predictor.")
    is_close = jnp.all(jnp.isclose(full_log_dens, reprod))
    assert_msg = "Serialized + deserialized predictor should produce the same results."
    assert is_close, assert_msg
    logger.info(
        "Assertion passed: the deserialized predictor produced the expected results."
    )

    est = mellon.DensityEstimator(rank=1.0, method="percent", n_landmarks=10)
    est.fit(X)
    dens_appr = est.predict(X)
    assert (
        relative_err(dens_appr) < 2e-1
    ), "The low landmarks approximation should be close to the default."

    # Test serialization
    est.predict.to_json(test_file)
    logger.info(f"Serialized the predictor and saved it to {test_file}.")
    predictor = mellon.Predictor.from_json(test_file)
    logger.info("Deserialized the predictor from the JSON file.")
    reprod = predictor(X)
    logger.info("Made a prediction with the deserialized predictor.")
    is_close = jnp.all(jnp.isclose(dens_appr, reprod))
    assert_msg = "Serialized + deserialized predictor should produce the same results."
    assert is_close, assert_msg
    logger.info(
        "Assertion passed: the deserialized predictor produced the expected results."
    )

    est = mellon.DensityEstimator(rank=0.99, method="percent", n_landmarks=80)
    est.fit(X)
    dens_appr = est.predict(X)
    assert (
        relative_err(dens_appr) < 2e-1
    ), "The low landmarks + Nystrom approximation should be close to the default."

    # Test serialization
    est.predict.to_json(test_file)
    logger.info(f"Serialized the predictor and saved it to {test_file}.")
    predictor = mellon.Predictor.from_json(test_file)
    logger.info("Deserialized the predictor from the JSON file.")
    reprod = predictor(X)
    logger.info("Made a prediction with the deserialized predictor.")
    is_close = jnp.all(jnp.isclose(dens_appr, reprod))
    assert_msg = "Serialized + deserialized predictor should produce the same results."
    assert is_close, assert_msg
    logger.info(
        "Assertion passed: the deserialized predictor produced the expected results."
    )

    # Test dictionay serialization
    data_dict = est.predict.to_dict()
    assert isinstance(data_dict, dict), "Predictor.to_dict() must return a dictionary."
    logger.info(f"Serialized the predictor to dictionary.")
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

    # Test compressed serialization
    est.predict.to_json(test_file, compress="gzip")
    logger.info(f"Serialized the predictor and saved it to {test_file}.")
    predictor = mellon.Predictor.from_json(test_file, compress="gzip")
    logger.info("Deserialized the predictor from the JSON file.")
    reprod = predictor(X)
    logger.info("Made a prediction with the deserialized predictor.")
    is_close = jnp.all(jnp.isclose(dens_appr, reprod))
    assert_msg = "Serialized + deserialized predictor with gzip compression should produce the same results."
    assert is_close, assert_msg
    logger.info(
        "Assertion passed: the deserialized predictor produced the expected results."
    )

    # Test compressed serialization
    est.predict.to_json(test_file, compress="bz2")
    logger.info(f"Serialized the predictor and saved it to {test_file}.")
    predictor = mellon.Predictor.from_json(test_file, compress="bz2")
    logger.info("Deserialized the predictor from the JSON file.")
    reprod = predictor(X)
    logger.info("Made a prediction with the deserialized predictor.")
    is_close = jnp.all(jnp.isclose(dens_appr, reprod))
    assert_msg = "Serialized + deserialized with bz2 compression predictor should produce the same results."
    assert is_close, assert_msg
    logger.info(
        "Assertion passed: the deserialized predictor produced the expected results."
    )

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


def test_TimeSensitiveDensityEstimator(tmp_path):
    n_per_batch = 10
    n_batches = 4
    test_time = 2
    n = n_per_batch * n_batches
    d = 2
    seed = 535
    test_file = tmp_path / "predictor.json"
    logger = mellon.Log()
    key = jax.random.PRNGKey(seed)
    L = jax.random.uniform(key, (d, d))
    cov = L.T.dot(L)
    X = jax.random.multivariate_normal(key, jnp.ones(d), cov, (n,))
    times = jnp.repeat(jnp.arange(n_batches), n_per_batch)

    est = mellon.TimeSensitiveDensityEstimator()
    log_dens = est.fit_predict(X, times)
    assert log_dens.shape == (n,), "There should be one density value for each sample."
    d_std = jnp.std(log_dens)

    def relative_err(dens):
        diff = jnp.std(log_dens - dens)
        return diff / d_std

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

    # Test serialization
    est.predict.to_json(test_file)
    logger.info(f"Serialized the predictor and saved it to {test_file}.")
    predictor = mellon.Predictor.from_json(test_file)
    logger.info("Deserialized the predictor from the JSON file.")
    reprod = predictor(X, times)
    logger.info("Made a prediction with the deserialized predictor.")
    is_close = jnp.all(jnp.isclose(pred_log_dens, reprod))
    assert_msg = "Serialized + deserialized predictor should produce the same results."
    assert is_close, assert_msg
    logger.info(
        "Assertion passed: the deserialized predictor produced the expected results."
    )

    est = mellon.TimeSensitiveDensityEstimator(
        rank=1.0, method="percent", n_landmarks=10
    )
    est.fit(X, times)
    dens_appr = est.predict(X, times)
    assert (
        relative_err(dens_appr) < 2e-1
    ), "The low landmarks approximation should be close to the default."

    # Test serialization
    est.predict.to_json(test_file)
    logger.info(f"Serialized the predictor and saved it to {test_file}.")
    predictor = mellon.Predictor.from_json(test_file)
    logger.info("Deserialized the predictor from the JSON file.")
    reprod = predictor(X, times)
    logger.info("Made a prediction with the deserialized predictor.")
    is_close = jnp.all(jnp.isclose(dens_appr, reprod))
    assert_msg = "Serialized + deserialized predictor should produce the same results."
    assert is_close, assert_msg
    logger.info(
        "Assertion passed: the deserialized predictor produced the expected results."
    )

    est = mellon.TimeSensitiveDensityEstimator(
        rank=0.99, method="percent", n_landmarks=80
    )
    est.fit(X, times)
    dens_appr = est.predict(X, times)
    assert (
        relative_err(dens_appr) < 5e-1
    ), "The low landmarks + Nystrom approximation should be close to the default."

    # Test serialization
    est.predict.to_json(test_file)
    logger.info(f"Serialized the predictor and saved it to {test_file}.")
    predictor = mellon.Predictor.from_json(test_file)
    logger.info("Deserialized the predictor from the JSON file.")
    reprod = predictor(X, times)
    logger.info("Made a prediction with the deserialized predictor.")
    is_close = jnp.all(jnp.isclose(dens_appr, reprod))
    assert_msg = "Serialized + deserialized predictor should produce the same results."
    assert is_close, assert_msg
    logger.info(
        "Assertion passed: the deserialized predictor produced the expected results."
    )


def test_FunctionEstimator():
    n = 100
    d = 2
    seed = 535
    key = jax.random.PRNGKey(seed)
    L = jax.random.uniform(key, (d, d))
    cov = L.T.dot(L)
    X = jax.random.multivariate_normal(key, jnp.ones(d), cov, (n,))
    noise = 1e-2 * jnp.sum(jnp.sin(X * 1e16), axis=1)
    noiseless_y = jnp.sum(jnp.sin(X / 2), axis=1)
    y = noiseless_y + noise
    Y = jnp.stack([y, noiseless_y])

    est = mellon.FunctionEstimator(sigma=1e-3)
    pred = est.fit_predict(X, y)
    assert pred.shape == (n,), "There should be a predicted value for each sample."

    assert len(str(est)) > 0, "The model should have a string representation."

    err = jnp.std(y - pred)
    assert err < 1e-2, "The prediction should be close to the intput value."

    err = jnp.std(noiseless_y - pred)
    assert err < 1e-2, "The prediction should be close to the true value."

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

    _ = est.multi_fit_predict(X, Y)
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

    m_pred = est.multi_fit_predict(X, Y)
    assert jnp.std(m_pred[0, :] - pred) < 1e-2, (
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
        relative_err(pred) < 1e-4
    ), "The predicive function should be consistent with the training samples."

    assert len(str(est)) > 0, "The model should have a string representation."

    adam_est = mellon.DimensionalityEstimator(optimizer="adam")
    adam_dim = adam_est.fit_predict(X)
    assert (
        relative_err(adam_dim) < 2e0
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

    grads = est.predict_density.gradient(X)
    assert (
        grads.shape == X.shape
    ), "The gradient should have the same shape as the input."

    log_dens = est.predict_density(X)
    hess = est.predict_density.hessian(X)
    assert hess.shape == (n, d, d), "The hessian should have the correct shape."

    result = est.predict_density.hessian_log_determinant(X)
    assert (
        len(result) == 2
    ), "hessian_log_determinan should return signes and lg-values."
    sng, ld = result
    assert sng.shape == (n,), "There should be one sign for each hessian determinan."
    assert ld.shape == (n,), "There should be one value for each hessian determinan."
