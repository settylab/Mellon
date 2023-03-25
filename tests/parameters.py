import mellon
import jax.numpy as jnp


def test_compute_mu():
    mu = mellon.compute_mu(jnp.arange(100), 4)
    assert hasattr(mu, "dtype")
    assert mu.dtype.kind == "f", "mu should be a floating point number."
    assert jnp.isfinite(mu), "mu should be finite."
    assert len(mu.shape) == 0, "mu should be scalar."


def test_compute_ls():
    ls = mellon.compute_ls(jnp.arange(1, 100))
    assert hasattr(ls, "dtype")
    assert ls.dtype.kind == "f", "ls should be a floating point number."
    assert jnp.isfinite(ls), "ls should be finite."
    assert len(ls.shape) == 0, "ls should be scalar."


def test_compute_cov_func():
    def test_curry(ls):
        def test_cov():
            return ls

        return test_cov

    test_ls = 10
    cov = mellon.compute_cov_func(test_curry, 10)
    assert callable(cov), "cov should be a callable function."
    assert cov() == test_ls, "cov should produce the expected value."


def test_compute_L():
    def cov(x, y):
        return jnp.ones((x.shape[0], y.shape[0]))

    n = 2
    d = 2
    X = jnp.ones((n, d))
    L = mellon.compute_L(X, cov)
    assert L.shape[0] == n, "L should have as many rows as there are samples."
    L = mellon.compute_L(X, cov, rank=1.0)
    assert L.shape == (n, n), "L should have full rank."
    L = mellon.compute_L(X, cov, rank=1)
    assert L.shape == (n, 1), "L should be reduced to rank == 1."
    mellon.compute_L(X, cov, rank=0.5)
    mellon.compute_L(X, cov, landmarks=X)
    L = mellon.compute_L(X, cov, landmarks=X, rank=1.0)
    assert L.shape == (n, n), "L should have full rank."
    L = mellon.compute_L(X, cov, landmarks=X, rank=1)
    assert L.shape == (n, 1), "L should be reduced to rank == 1."
    mellon.compute_L(X, cov, landmarks=X, rank=0.5)


def test_compute_initial_value():
    n = 2
    d = 2
    iv = mellon.compute_initial_value(jnp.arange(n) + 1, 3, 1, jnp.ones((n, d)))
    assert iv.dtype.kind == "f", "The initial value should have floating point numbers."
    assert iv.shape == (d,), "The initial value should have the right dimensionality."
    assert jnp.isfinite(iv).all(), "The initial value should have finite entries."
