import jax.numpy as jnp
import mellon


def test_Matern32():
    n = 2
    d = 2
    cov = mellon.cov.Matern32(1.2)
    assert (
        len(str(cov)) > 0
    ), "The covariance function should have a string representation."
    x = jnp.ones((n, d))
    values = cov(x, 2 * x)
    assert values.shape == (
        d,
        d,
    ), "Covariance should be computed for each pair of samples."


def test_Matern52():
    n = 2
    d = 2
    cov = mellon.cov.Matern52(1.2)
    assert (
        len(str(cov)) > 0
    ), "The covariance function should have a string representation."
    x = jnp.ones((n, d))
    values = cov(x, 2 * x)
    assert values.shape == (
        d,
        d,
    ), "Covariance should be computed for each pair of samples."


def test_ExpQuad():
    n = 2
    d = 2
    cov = mellon.cov.ExpQuad(1.2)
    assert (
        len(str(cov)) > 0
    ), "The covariance function should have a string representation."
    x = jnp.ones((n, d))
    values = cov(x, 2 * x)
    assert values.shape == (
        d,
        d,
    ), "Covariance should be computed for each pair of samples."


def test_Exponential():
    n = 2
    d = 2
    cov = mellon.cov.Exponential(1.2)
    assert (
        len(str(cov)) > 0
    ), "The covariance function should have a string representation."
    x = jnp.ones((n, d))
    values = cov(x, 2 * x)
    assert values.shape == (
        d,
        d,
    ), "Covariance should be computed for each pair of samples."


def test_RatQuad():
    n = 2
    d = 2
    cov = mellon.cov.RatQuad(3, 1.2)
    assert (
        len(str(cov)) > 0
    ), "The covariance function should have a string representation."
    x = jnp.ones((n, d))
    values = cov(x, 2 * x)
    assert values.shape == (
        d,
        d,
    ), "Covariance should be computed for each pair of samples."
