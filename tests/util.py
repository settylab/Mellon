import mellon
import jax
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

def test_test_rank():
    seed = 423
    shape = (5, 10)
    key = jax.random.PRNGKey(seed)
    L = jax.random.uniform(key, shape=shape)

    mellon.test_rank(L)
    rank = mellon.test_rank(L, tol=.5)
    assert rank == 4, "The approx. rank with tol=.5 of the test matrix should be 4."
    mellon.test_rank(L, threshold=.5)
    mellon.test_rank(L, tol=1, threshold=.5)

    est = mellon.DensityEstimator().fit(L)
    rank = mellon.test_rank(est)
    assert rank == 1, "The approx. rank of the test data should be 1."

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
