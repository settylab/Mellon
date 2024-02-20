import jax
import jax.numpy as jnp
import mellon.cov
import pytest

# Define the covariance classes to be tested
COVARIANCE_CLASSES = [
    mellon.cov.Matern32,
    mellon.cov.Matern52,
    mellon.cov.ExpQuad,
    mellon.cov.Exponential,
    mellon.cov.RatQuad,
    mellon.cov.Linear,
]

# Define the active dimensions to be tested
ACTIVE_DIMS = [None, slice(2), 1, slice(None, None, 2), [1, 2]]


# Parametrize over both covariance classes and active dimensions
@pytest.mark.parametrize(
    "CovarianceClass, active_dims",
    [(cls, ad) for cls in COVARIANCE_CLASSES for ad in ACTIVE_DIMS],
)
def test_covariance_class(CovarianceClass, active_dims):
    n, d = 5, 4
    ls = 1.2

    # Instantiate the covariance class with active_dims
    if CovarianceClass.__name__ == "RatQuad":
        cov = CovarianceClass(3, ls, active_dims=active_dims)
    else:
        cov = CovarianceClass(ls, active_dims=active_dims)

    # Basic string representation check
    assert (
        len(str(cov)) > 0
    ), "The covariance function should have a string representation."

    # Set up sample points
    x = jnp.ones((n, d))
    y = jnp.ones((n + 1, d)) * 2
    y = y.at[0].set(2)
    y = y.at[1].set(1.5)

    # Check covariance shape
    values = cov(x, y)
    assert values.shape == (
        n,
        n + 1,
    ), "Covariance should be computed for each pair of samples."

    # Compute the gradient using k_grad
    k_grad_func = cov.k_grad(x)
    computed_grad = k_grad_func(y)

    # Compute the gradient using JAX automatic differentiation
    k_func = lambda y: cov.k(x, y[None,])[..., 0]
    expected_grad = jax.vmap(jax.jacfwd(k_func), in_axes=(0,), out_axes=1)(y)

    # Assert that the gradients are close
    assert jnp.allclose(
        computed_grad, expected_grad, atol=1e-6
    ), f"Gradients do not match in {CovarianceClass.__name__} with active_dims {active_dims}"
