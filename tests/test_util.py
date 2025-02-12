import logging
import mellon
import jax
import jax.numpy as jnp


def test_mle():
    nn_distances = jnp.arange(1, 10)
    dens = mellon.util.mle(nn_distances, 2)
    assert (
        dens.shape == nn_distances.shape
    ), "ML estimate should produce one value for each input."


def test_distances():
    n = 2
    x = jnp.ones((n, 3))
    dist = mellon.util.distance(x, x)
    assert dist.shape == (n, n), "Distances should be computed for each pair of points."


def test_distance_grad_shapes():
    x = jnp.array([[0, 0], [1, 1]])
    y = jnp.array([[1, 0], [0, 1]])
    dist_grad_func = mellon.util.distance_grad(x)
    distance, gradient = dist_grad_func(y)

    assert distance.shape == (2, 2), "Distance shape is incorrect"
    assert gradient.shape == (2, 2, 2), "Gradient shape is incorrect"


def test_distance_grad():
    n = 4
    d = 2
    x = jnp.ones((n, d))
    y = jnp.ones((n - 1, d)) * 2
    y = y.at[0].set(2)
    y = y.at[1].set(1.5)

    dist_grad_f = mellon.util.distance_grad(x)
    distance, computed_grad = dist_grad_f(y)

    expected_distance = mellon.util.distance(x, y)
    assert jnp.allclose(
        distance, expected_distance, atol=1e-6
    ), "Distances do not match"

    # Compute the gradient using JAX automatic differentiation
    k_func = lambda y: mellon.util.distance(x, y[None,])[..., 0]
    expected_grad = jax.vmap(jax.jacfwd(k_func), in_axes=(0,))(y)
    expected_grad = expected_grad.transpose([1, 0, 2])

    # Assert that the gradients are close
    assert jnp.allclose(
        computed_grad, expected_grad, atol=1e-6
    ), "Gradients do not match"


def test_test_rank():
    seed = 423
    key = jax.random.PRNGKey(seed)

    # Define a controlled singular value spectrum
    shape = (5, 10)
    singular_values = jnp.array([3.0, 2.0, 1.5, 1.0, 0.4])  # Controlled decay
    U, _ = jnp.linalg.qr(
        jax.random.normal(key, (shape[0], shape[0]))
    )  # Orthogonal matrix
    V, _ = jnp.linalg.qr(
        jax.random.normal(key, (shape[1], shape[1]))
    )  # Orthogonal matrix

    # Construct matrix with controlled singular values
    S = jnp.zeros((shape[0], shape[1]))
    S = S.at[: len(singular_values), : len(singular_values)].set(
        jnp.diag(singular_values)
    )
    L = U @ S @ V.T  # Controlled rank approximation

    mellon.util.test_rank(L)
    rank = mellon.util.test_rank(L, tol=0.5)
    assert rank == 4, "The approx. rank with tol=0.5 of the test matrix should be 4."

    mellon.util.test_rank(L, threshold=0.5)
    mellon.util.test_rank(L, tol=1, threshold=0.5)

    est = mellon.DensityEstimator().fit(L)
    rank = mellon.util.test_rank(est)
    assert rank == 1, "The approx. rank of the test data should be 1."


def test_local_dimensionality():
    n = 10
    x = jnp.ones((n, 3))
    dist = mellon.util.local_dimensionality(x, k=3)
    assert dist.shape == (n,), "Local dim should be computed for each point."


def test_set_verbosity_to_false_changes_level_to_warning(caplog):
    mellon.util.set_verbosity(False)
    assert (
        mellon.logger.getEffectiveLevel() == logging.WARNING
    ), "Logging level should be set to WARNING when verbosity is False."


def test_set_verbosity_to_true_changes_level_to_info(caplog):
    mellon.util.set_verbosity(True)
    assert (
        mellon.logger.getEffectiveLevel() == logging.INFO
    ), "Logging level should be set to INFO when verbosity is True."


def test_set_jax_config():
    mellon.util.set_jax_config()
