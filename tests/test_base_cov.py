import pytest
import jax
import jax.numpy as jnp
import mellon

# Define the active dimensions to be tested
ACTIVE_DIMS = [None, slice(2), 1, slice(None, None, 2)]


@pytest.mark.parametrize(
    "active_dims",
    ACTIVE_DIMS,
)
def test_Add(active_dims):
    n = 2
    d = 3
    cov1 = mellon.cov.Matern32(1.4)
    cov2 = mellon.cov.Exponential(3.4)

    cov = cov1 + cov2
    assert (
        len(str(cov)) > 0
    ), "The covariance function should have a string representation."
    x = jnp.ones((n, d))
    values = cov(x, 2 * x)
    assert values.shape == (
        n,
        n,
    ), "Covariance should be computed for each pair of samples."

    cov.active_dims = active_dims
    values = cov(x, 2 * x)

    json = cov.to_json()
    recov = mellon.cov.Covariance.from_json(json)
    revalues = recov(x, 2 * x)
    assert jnp.all(
        jnp.isclose(values, revalues)
    ), "Serialization + deserialization of added covariance functions must return the same result."

    # Compute the gradient using k_grad
    y = 2 * x
    k_grad_func = cov.k_grad(x)
    computed_grad = k_grad_func(y)

    # Compute the gradient using JAX automatic differentiation
    k_func = lambda y: cov.k(x, y[None,])[..., 0]
    expected_grad = jax.vmap(jax.jacfwd(k_func), in_axes=(0,), out_axes=1)(y)

    # Assert that the gradients are close
    assert jnp.allclose(
        computed_grad, expected_grad, atol=1e-6
    ), f"Gradients do not match in {cov.__class__.__name__} covariance with active_dims {active_dims}"


@pytest.mark.parametrize(
    "active_dims",
    ACTIVE_DIMS,
)
def test_Mul(active_dims):
    n = 2
    d = 3
    cov1 = mellon.cov.Matern32(1.4)
    cov2 = mellon.cov.Exponential(3.4)

    cov = cov1 * cov2
    assert (
        len(str(cov)) > 0
    ), "The covariance function should have a string representation."
    x = jnp.ones((n, d))
    values = cov(x, 2 * x)
    assert values.shape == (
        n,
        n,
    ), "Covariance should be computed for each pair of samples."

    cov.active_dims = active_dims
    values = cov(x, 2 * x)

    json = cov.to_json()
    recov = mellon.cov.Covariance.from_json(json)
    revalues = recov(x, 2 * x)
    assert jnp.all(
        jnp.isclose(values, revalues)
    ), "Serialization + deserialization of added covariance functions must return the same result."

    # Compute the gradient using k_grad
    y = 2 * x
    k_grad_func = cov.k_grad(x)
    computed_grad = k_grad_func(y)

    # Compute the gradient using JAX automatic differentiation
    k_func = lambda y: cov.k(x, y[None,])[..., 0]
    expected_grad = jax.vmap(jax.jacfwd(k_func), in_axes=(0,), out_axes=1)(y)

    # Assert that the gradients are closn
    assert jnp.allclose(
        computed_grad, expected_grad, atol=1e-6
    ), f"Gradients do not match in {cov.__class__.__name__} covariance with active_dims {active_dims}"


@pytest.mark.parametrize(
    "active_dims",
    ACTIVE_DIMS,
)
def test_Pow(active_dims):
    n = 2
    d = 3
    cov1 = mellon.cov.Matern32(1.4)

    cov = cov1**3.2
    assert (
        len(str(cov)) > 0
    ), "The covariance function should have a string representation."
    x = jnp.ones((n, d))
    values = cov(x, 2 * x)
    assert values.shape == (
        n,
        n,
    ), "Covariance should be computed for each pair of samples."

    cov.active_dims = active_dims
    values = cov(x, 2 * x)

    json = cov.to_json()
    recov = mellon.cov.Covariance.from_json(json)
    revalues = recov(x, 2 * x)
    assert jnp.all(
        jnp.isclose(values, revalues)
    ), "Serialization + deserialization of added covariance functions must return the same result."

    # Compute the gradient using k_grad
    y = 2 * x
    k_grad_func = cov.k_grad(x)
    computed_grad = k_grad_func(y)

    # Compute the gradient using JAX automatic differentiation
    k_func = lambda y: cov.k(x, y[None,])[..., 0]
    expected_grad = jax.vmap(jax.jacfwd(k_func), in_axes=(0,), out_axes=1)(y)

    # Assert that the gradients are close
    assert jnp.allclose(
        computed_grad, expected_grad, atol=1e-6
    ), f"Gradients do not match in {cov.__class__.__name__} covariance with active_dims {active_dims}"


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

    # Compute the gradient using k_grad
    y = 2 * x**2
    k_grad_func = cov.k_grad(x)
    computed_grad = k_grad_func(y)

    # Compute the gradient using JAX automatic differentiation
    k_func = lambda y: cov.k(x, y[None,])[..., 0]
    expected_grad = jax.vmap(jax.jacfwd(k_func), in_axes=(0,), out_axes=1)(y)

    # Assert that the gradients are close
    assert jnp.allclose(
        computed_grad, expected_grad, atol=1e-6
    ), f"Gradients do not match in hirachichal covariance."
