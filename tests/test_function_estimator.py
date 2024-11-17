import pytest
import mellon
import jax
import jax.numpy as jnp


@pytest.fixture
def function_estimator_setup():
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
    Y = jnp.stack([y, noiseless_y], axis=1)

    return X, y, Y, noiseless_y


def test_function_estimator_prediction(function_estimator_setup):
    X, y, _, noiseless_y = function_estimator_setup
    n = X.shape[0]

    with pytest.raises(ValueError):
        mellon.FunctionEstimator(gp_type="sparse_nystroem")

    est = mellon.FunctionEstimator(sigma=1e-3)

    with pytest.raises(ValueError):
        est.fit_predict()

    pred = est.fit_predict(X, y)

    html_output = est._repr_html_()
    str_output = str(est)

    assert pred.shape == (n,), "There should be a predicted value for each sample."
    assert len(str(est)) > 0, "The model should have a string representation."
    err = jnp.std(y - pred)
    assert err < 1e-2, "The prediction should be close to the input value."
    err = jnp.std(noiseless_y - pred)
    assert err < 1e-2, "The prediction should be close to the true value."

    pred_self = est(X, y)
    assert jnp.all(
        jnp.isclose(pred, pred_self)
    ), "__call__() shoud return the same as predict()"

    est.compute_conditional(y=y)
    est.compute_conditional(x=y, y=y)

    with pytest.raises(ValueError):
        est.compute_conditional(X)

    with pytest.raises(ValueError):
        est.fit(X, y[:3])

    with pytest.raises(ValueError):
        est.fit_predict(X[:, :, None], y)

    with pytest.raises(ValueError):
        est.fit_predict(X[:3, :], y)


def test_function_estimator_multi_fit_predict(function_estimator_setup):
    X, y, Y, _ = function_estimator_setup
    n = X.shape[0]
    est = mellon.FunctionEstimator(sigma=1e-3)

    m_pred = est.fit_predict(X, Y, X)
    assert m_pred.shape == (
        n,
        2,
    ), "There should be a value for each sample and location."
    est.multi_fit_predict(X, Y.T, X)


@pytest.mark.parametrize("n_landmarks, error_limit", [(0, 1e-4), (10, 1e-1)])
def test_function_estimator_approximations(
    function_estimator_setup, n_landmarks, error_limit
):
    X, y, _, _ = function_estimator_setup
    est_default = mellon.FunctionEstimator(sigma=1e-3)
    pred_default = est_default.fit_predict(X, y)

    est = mellon.FunctionEstimator(sigma=1e-3, n_landmarks=n_landmarks)
    pred_appr = est.fit_predict(X, y)

    err = jnp.std(pred_appr - pred_default)
    assert err < error_limit, "The approximation should be close to the default."


@pytest.mark.parametrize("n_landmarks, error_limit", [(0, 1e-5), (10, 4e-1)])
def test_function_estimator_approximations_1d(
    function_estimator_setup, n_landmarks, error_limit
):
    X, y, _, _ = function_estimator_setup
    est_default = mellon.FunctionEstimator(sigma=1e-3)
    d1_pred_default = est_default.fit_predict(X[:, 0], y)

    est = mellon.FunctionEstimator(sigma=1e-3, n_landmarks=n_landmarks)
    d1_pred = est.fit_predict(X[:, 0], y)

    assert (
        jnp.std(d1_pred - d1_pred_default) < error_limit
    ), "The scalar state function estimations should be consistent under approximation."
