import pytest
from enum import Enum
import jax.numpy as jnp
import mellon
from mellon.parameters import (
    compute_n_landmarks,
    compute_rank,
    GaussianProcessType,
    compute_gp_type,
)


def test_gaussian_process_type():
    assert (
        GaussianProcessType.from_string("full") == GaussianProcessType.FULL
    ), "Error converting 'full' to FULL enum."
    assert (
        GaussianProcessType.from_string("full_nystroem")
        == GaussianProcessType.FULL_NYSTROEM
    ), "Error converting 'full_nystroem' to FULL_NYSTROEM enum."
    assert (
        GaussianProcessType.from_string("sparse_cholesky")
        == GaussianProcessType.SPARSE_CHOLESKY
    ), "Error converting 'sparse_cholesky' to SPARSE_CHOLESKY enum."
    assert (
        GaussianProcessType.from_string("sparse_nystroem")
        == GaussianProcessType.SPARSE_NYSTROEM
    ), "Error converting 'sparse_nystroem' to SPARSE_NYSTROEM enum."

    with pytest.raises(ValueError):
        GaussianProcessType.from_string(
            "unknown_type"
        ), "Error was expected with unknown Gaussian Process type."

    partial_input = GaussianProcessType.from_string("sparse")
    assert isinstance(
        partial_input, GaussianProcessType
    ), "Error converting partial input to an enum instance."
    assert partial_input in [
        GaussianProcessType.SPARSE_CHOLESKY,
        GaussianProcessType.SPARSE_NYSTROEM,
    ], "Error matching partial input to one of the SPARSE enums."

    none_input = GaussianProcessType.from_string(None, optional=True)
    assert none_input is None, "Error handling None input with optional flag."

    with pytest.raises(ValueError):
        GaussianProcessType.from_string(
            None
        ), "Error was expected with None input without optional flag."


def test_compute_gp_type():
    # Test full model with integer rank, float rank, None rank, and zero rank
    assert compute_gp_type(0, 100, 100) == GaussianProcessType.FULL
    assert compute_gp_type(100, 1.0, 100) == GaussianProcessType.FULL
    assert compute_gp_type(100, None, 100) == GaussianProcessType.FULL
    assert compute_gp_type(100, 0, 100) == GaussianProcessType.FULL

    # Test full model with Nyström rank reduction
    assert compute_gp_type(100, 50, 100) == GaussianProcessType.FULL_NYSTROEM
    assert compute_gp_type(100, 0.5, 100) == GaussianProcessType.FULL_NYSTROEM

    # Test sparse model with integer rank, float rank, None rank, and zero rank
    assert compute_gp_type(50, 50, 100) == GaussianProcessType.SPARSE_CHOLESKY
    assert compute_gp_type(50, 1.0, 100) == GaussianProcessType.SPARSE_CHOLESKY
    assert compute_gp_type(50, None, 100) == GaussianProcessType.SPARSE_CHOLESKY
    assert compute_gp_type(50, 0, 100) == GaussianProcessType.SPARSE_CHOLESKY

    # Test sparse model with Nyström rank reduction
    assert compute_gp_type(50, 25, 100) == GaussianProcessType.SPARSE_NYSTROEM
    assert compute_gp_type(50, 0.5, 100) == GaussianProcessType.SPARSE_NYSTROEM


def test_compute_mu():
    mu = mellon.parameters.compute_mu(jnp.arange(100), 4)
    assert isinstance(mu, float)
    assert jnp.isfinite(mu), "mu should be finite."


def test_compute_ls():
    ls = mellon.parameters.compute_ls(jnp.arange(1, 100))
    assert isinstance(ls, float)
    assert jnp.isfinite(ls), "ls should be finite."


def test_compute_cov_func():
    def test_curry(ls):
        def test_cov():
            return ls

        return test_cov

    test_ls = 10
    cov = mellon.parameters.compute_cov_func(test_curry, 10)
    assert callable(cov), "cov should be a callable function."
    assert cov() == test_ls, "cov should produce the expected value."


def test_compute_rank():
    assert compute_rank(GaussianProcessType.FULL_NYSTROEM) == 0.99
    assert compute_rank(GaussianProcessType.SPARSE_CHOLESKY) == 1.0
    assert compute_rank(None) == 1.0


def test_compute_n_landmarks():
    DEFAULT_N_LANDMARKS = 5000

    # Test when landmarks are not None
    landmarks = jnp.ones((50, 2))
    assert compute_n_landmarks(None, 100, landmarks) == 50

    # Test when gp_type is None
    assert compute_n_landmarks(None, 100, None) == min(100, DEFAULT_N_LANDMARKS)

    # Test with FULL or FULL_NYSTROEM gp_type
    assert compute_n_landmarks(GaussianProcessType.FULL, 100, None) == 100
    assert compute_n_landmarks(GaussianProcessType.FULL_NYSTROEM, 100, None) == 100

    # Test with SPARSE_CHOLESKY or SPARSE_NYSTROEM gp_type
    assert (
        compute_n_landmarks(GaussianProcessType.SPARSE_CHOLESKY, 100, None)
        == DEFAULT_N_LANDMARKS
    )
    assert (
        compute_n_landmarks(GaussianProcessType.SPARSE_NYSTROEM, 100, None)
        == DEFAULT_N_LANDMARKS
    )

    # Test with SPARSE_CHOLESKY or SPARSE_NYSTROEM gp_type and n_samples <= DEFAULT_N_LANDMARKS
    assert (
        compute_n_landmarks(GaussianProcessType.SPARSE_CHOLESKY, 80, None)
        == DEFAULT_N_LANDMARKS
    )
    assert (
        compute_n_landmarks(GaussianProcessType.SPARSE_NYSTROEM, 80, None)
        == DEFAULT_N_LANDMARKS
    )

    # Test with unknown gp_type
    class UnknownType(Enum):
        UNKNOWN = "unknown"

    assert compute_n_landmarks(UnknownType.UNKNOWN, 100, None) == min(
        100, DEFAULT_N_LANDMARKS
    )


def test_compute_L():
    def cov(x, y):
        return jnp.ones((x.shape[0], y.shape[0]))

    n = 2
    d = 2
    X = jnp.ones((n, d))
    L = mellon.parameters.compute_L(X, cov)
    assert L.shape[0] == n, "L should have as many rows as there are samples."
    L = mellon.parameters.compute_L(X, cov, rank=1.0)
    assert L.shape == (n, n), "L should have full rank."
    L = mellon.parameters.compute_L(X, cov, rank=1)
    assert L.shape == (n, 1), "L should be reduced to rank == 1."
    mellon.parameters.compute_L(X, cov, rank=0.5)
    mellon.parameters.compute_L(X, cov, landmarks=X)
    L = mellon.parameters.compute_L(X, cov, landmarks=X, rank=1.0)
    assert L.shape == (n, n), "L should have full rank."
    L = mellon.parameters.compute_L(X, cov, landmarks=X, rank=1)
    assert L.shape == (n, 1), "L should be reduced to rank == 1."
    mellon.parameters.compute_L(X, cov, landmarks=X, rank=0.5)


def test_compute_initial_value():
    n = 2
    d = 2
    iv = mellon.parameters.compute_initial_value(
        jnp.arange(n) + 1, 3, 1, jnp.ones((n, d))
    )
    assert iv.dtype.kind == "f", "The initial value should have floating point numbers."
    assert iv.shape == (d,), "The initial value should have the right dimensionality."
    assert jnp.isfinite(iv).all(), "The initial value should have finite entries."
