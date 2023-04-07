import mellon
import jax.numpy as jnp


def test_stabilize():
    A = jnp.ones((2, 2))
    B = mellon.stabilize(A)
    assert A.shape == B.shape, "Stabilization should conserve shape."
    jitter = 1e-6
    B = mellon.stabilize(A, jitter=jitter)
    assert (
        jnp.max(jnp.abs(A - B)) <= jitter
    ), "The stabilized matrix should not deviate too much."


def test_mle():
    nn_distances = jnp.arange(1, 10)
    dens = mellon.mle(nn_distances, 2)
    assert (
        dens.shape == nn_distances.shape
    ), "ML estimate should produce one value for each input."


def test_distances():
    n = 2
    x = jnp.ones((n, 3))
    dist = mellon.distance(x, x)
    assert dist.shape == (n, n), "Distances should be computed for each pair of points."


def test_local_dimensionality():
    n = 10
    x = jnp.ones((n, 3))
    dist = mellon.local_dimensionality(x, k=3)
    assert dist.shape == (n,), "Local dim should be computed for each point."


def test_Log():
    logger = mellon.Log()
    assert logger is mellon.Log(), "Log should be a singelton class."
    assert hasattr(logger, "debug")
    assert hasattr(logger, "info")
    assert hasattr(logger, "warn")
    assert hasattr(logger, "error")
