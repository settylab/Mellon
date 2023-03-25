import jax.numpy as jnp
import mellon


def test_Add():
    n = 2
    d = 2
    cov1 = mellon.Matern32(1.4)
    cov2 = mellon.Exponential(3.4)

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


def test_Mul():
    n = 2
    d = 2
    cov1 = mellon.Matern32(1.4)
    cov2 = mellon.Exponential(3.4)

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


def test_Pow():
    n = 2
    d = 2
    cov1 = mellon.Matern32(1.4)

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
