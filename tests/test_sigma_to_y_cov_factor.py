import jax.numpy as jnp
import pytest
from mellon.conditional import _sigma_to_y_cov_factor


def test_scalar_sigma():
    sigma = 0.5
    n = 3
    expected = jnp.eye(n) * sigma
    result = _sigma_to_y_cov_factor(sigma, None, n)
    assert jnp.allclose(result, expected), "Failed for scalar sigma"


def test_vector_sigma():
    sigma = jnp.array([1.0, 2.0, 3.0])
    n = 3
    expected = jnp.diag(sigma)
    result = _sigma_to_y_cov_factor(sigma, None, n)
    assert jnp.allclose(result, expected), "Failed for vector sigma"


def test_higher_dimensional_sigma():
    sigma = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    n = 2
    expected = jnp.array([[[1.0, 2.0], [0.0, 0.0]], [[0.0, 0.0], [3.0, 4.0]]])
    result = _sigma_to_y_cov_factor(sigma, None, n)
    assert jnp.allclose(result, expected), "Failed for higher-dimensional sigma"


def test_both_sigma_y_cov_factor_provided():
    sigma = jnp.array([1.0, 2.0, 3.0])
    y_cov_factor = jnp.eye(3)
    n = 3
    with pytest.raises(ValueError):
        _sigma_to_y_cov_factor(sigma, y_cov_factor, n)


def test_neither_sigma_nor_y_cov_factor_provided():
    with pytest.raises(ValueError):
        _sigma_to_y_cov_factor(None, None, 3)
