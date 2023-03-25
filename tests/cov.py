import jax.numpy as jnp
import mellon


def test_Matern32():
    n = 2
    d = 2
    cov = mellon.Matern32(1.2)
    assert (
        len(str(cov)) > 0
    ), "The covariance function should have a string representation."
    x = jnp.ones((n, d))
    values = cov.k(x, 2 * x)
    assert values.shape == (
        d,
        d,
    ), "Covariance should be computed for each pair of samples."


def test_Matern52():
    n = 2
    d = 2
    cov = mellon.Matern52(1.2)
    assert (
        len(str(cov)) > 0
    ), "The covariance function should have a string representation."
    x = jnp.ones((n, d))
    values = cov.k(x, 2 * x)
    assert values.shape == (
        d,
        d,
    ), "Covariance should be computed for each pair of samples."


def test_ExpQuad():
    n = 2
    d = 2
    cov = mellon.ExpQuad(1.2)
    assert (
        len(str(cov)) > 0
    ), "The covariance function should have a string representation."
    x = jnp.ones((n, d))
    values = cov.k(x, 2 * x)
    assert values.shape == (
        d,
        d,
    ), "Covariance should be computed for each pair of samples."


def test_Exponential():
    n = 2
    d = 2
    cov = mellon.Exponential(1.2)
    assert (
        len(str(cov)) > 0
    ), "The covariance function should have a string representation."
    x = jnp.ones((n, d))
    values = cov.k(x, 2 * x)
    assert values.shape == (
        d,
        d,
    ), "Covariance should be computed for each pair of samples."


def test_RatQuad():
    n = 2
    d = 2
    cov = mellon.RatQuad(3, 1.2)
    assert (
        len(str(cov)) > 0
    ), "The covariance function should have a string representation."
    x = jnp.ones((n, d))
    values = cov.k(x, 2 * x)
    assert values.shape == (
        d,
        d,
    ), "Covariance should be computed for each pair of samples."
