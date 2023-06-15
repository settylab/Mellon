import jax.numpy as jnp
import mellon


def test_Add():
    n = 2
    d = 2
    cov1 = mellon.cov.Matern32(1.4)
    cov2 = mellon.cov.Exponential(3.4)

    cov = cov1 + cov2
    assert (
        len(str(cov)) > 0
    ), "The covariance function should have a string representation."
    x = jnp.ones((n, d))
    values = cov(x, 2 * x)
    assert values.shape == (
        d,
        d,
    ), "Covariance should be computed for each pair of samples."

    json = cov.to_json()
    recov = mellon.cov.Covariance.from_json(json)
    revalues = recov(x, 2 * x)
    assert jnp.all(
        jnp.isclose(values, revalues)
    ), "Serialization + deserialization of added covariance functions must return the same result."


def test_Mul():
    n = 2
    d = 2
    cov1 = mellon.cov.Matern32(1.4)
    cov2 = mellon.cov.Exponential(3.4)

    cov = cov1 * cov2
    assert (
        len(str(cov)) > 0
    ), "The covariance function should have a string representation."
    x = jnp.ones((n, d))
    values = cov(x, 2 * x)
    assert values.shape == (
        d,
        d,
    ), "Covariance should be computed for each pair of samples."

    json = cov.to_json()
    recov = mellon.cov.Covariance.from_json(json)
    revalues = recov(x, 2 * x)
    assert jnp.all(
        jnp.isclose(values, revalues)
    ), "Serialization + deserialization of added covariance functions must return the same result."


def test_Pow():
    n = 2
    d = 2
    cov1 = mellon.cov.Matern32(1.4)

    cov = cov1**3.2
    assert (
        len(str(cov)) > 0
    ), "The covariance function should have a string representation."
    x = jnp.ones((n, d))
    values = cov(x, 2 * x)
    assert values.shape == (
        d,
        d,
    ), "Covariance should be computed for each pair of samples."

    json = cov.to_json()
    recov = mellon.cov.Covariance.from_json(json)
    revalues = recov(x, 2 * x)
    assert jnp.all(
        jnp.isclose(values, revalues)
    ), "Serialization + deserialization of added covariance functions must return the same result."


def test_Hirachical():
    n = 2
    d = 3
    cov1 = mellon.cov.Matern52(1.4, active_dims=0)
    cov2 = mellon.cov.Exponential(3.4, active_dims=[1, 2])
    cov3 = mellon.cov.RatQuad(1.1, 3.4, active_dims=slice(0, 2, 1))
    cov4 = mellon.cov.Matern52(1.0, active_dims=[False, True, True])

    cov = 0.2 + 1.1 * cov1 + 2.1 * cov2 * cov3 + cov4
    assert (
        len(str(cov)) > 0
    ), "The covariance function should have a string representation."
    x = jnp.ones((n, d))
    values = cov(x, 2 * x)
    assert values.shape == (
        n,
        n,
    ), "Covariance should be computed for each pair of samples."

    json = cov.to_json()
    recov = mellon.cov.Covariance.from_json(json)
    revalues = recov(x, 2 * x)
    assert jnp.all(
        jnp.isclose(values, revalues)
    ), "Serialization + deserialization of added covariance functions must return the same result."
