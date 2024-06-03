import pytest
from enum import Enum
import jax.numpy as jnp
from jax import random
import logging
import mellon
from mellon.parameters import (
    compute_n_landmarks,
    compute_rank,
    compute_nn_distances,
    compute_gp_type,
    compute_landmarks_rescale_time,
    compute_nn_distances_within_time_points,
    compute_d,
    compute_d_factal,
    compute_Lp,
    compute_L,
)
from mellon.util import GaussianProcessType


def test_compute_landmarks_rescale_time():
    x = jnp.array([[1, 2], [3, 4], [3, 5]])

    lm = compute_landmarks_rescale_time(x, 1, 1, n_landmarks=0)
    assert lm is None, "Non should be returned if n_landmarks=0"

    # Testing input validation by passing negative length scales
    with pytest.raises(ValueError):
        compute_landmarks_rescale_time(x, -1, 1)

    with pytest.raises(ValueError):
        compute_landmarks_rescale_time(x, 1, -1)

    lm = compute_landmarks_rescale_time(x, 1, 2, n_landmarks=2)
    assert lm.shape == (2, 2), "`n_landmarks` landmars should be retuned."


def test_compute_nn_distances_within_time_points():
    # Test basic functionality without normalization
    x = jnp.array([[1, 2, 0], [3, 4, 0], [5, 6, 1], [7, 8, 1]])
    result = compute_nn_distances_within_time_points(x)
    assert result.shape == (4,)

    # Test behavior with insufficient samples at a given time point
    x_single_sample = jnp.array([[1, 2, 0], [3, 4, 1], [5, 6, 2]])
    with pytest.raises(ValueError):
        compute_nn_distances_within_time_points(x_single_sample)

    # Test functionality with the times array passed separately
    x_without_times = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    times = jnp.array([0, 0, 1, 1])
    result_with_times = compute_nn_distances_within_time_points(
        x_without_times, times=times
    )
    assert jnp.all(result_with_times == result)


def test_compute_d_factal(caplog):
    # Create a random key for jax.random
    key = random.PRNGKey(0)

    # Create a random array using jax.random
    x_2d = random.normal(key, shape=(100, 10))
    result_2d = compute_d_factal(x_2d)
    assert isinstance(result_2d, float)

    # Test with 1D array (should return 1)
    x_1d = random.normal(key, shape=(100,))
    assert compute_d_factal(x_1d) == 1

    # Test with k > number of samples (expect a warning)
    x_small = random.normal(key, shape=(5, 10))
    logger = logging.getLogger("mellon")
    logger.propagate = True
    with caplog.at_level(logging.WARNING, logger="mellon"):
        compute_d_factal(x_small, k=10)
    logger.propagate = False
    assert "is greater than the number of samples" in caplog.text

    # Test with specific random seed
    x_seed = random.normal(key, shape=(100, 10))
    result_seed = compute_d_factal(x_seed, seed=432)
    assert isinstance(result_seed, float)

    # Test with n < number of samples
    x_n = random.normal(key, shape=(1000, 10))
    result_n = compute_d_factal(x_n, n=500)
    assert isinstance(result_n, float)

    # Test with invalid input (negative k)
    with pytest.raises(ValueError):
        compute_d_factal(x_2d, k=-5)


def test_compute_Lp():
    # Generate some mock data and landmarks
    x = jnp.array([[1, 2], [3, 4], [5, 6]])
    landmarks = jnp.array([[1, 2], [3, 4]])
    mock_cov_func = mellon.cov.Matern52(1)

    # Test 'full' Gaussian Process type
    Lp = compute_Lp(x, mock_cov_func, gp_type="full")
    assert Lp.shape == (3, 3)
    assert isinstance(Lp, jnp.ndarray)

    # Test 'sparse_cholesky' with landmarks
    Lp_sparse = compute_Lp(
        x, mock_cov_func, gp_type="sparse_cholesky", landmarks=landmarks
    )
    assert Lp_sparse.shape == (2, 2)
    assert isinstance(Lp_sparse, jnp.ndarray)

    # Test full Nyström should return None
    assert compute_Lp(x, mock_cov_func, gp_type="full_nystroem") is None

    # Test sparse Nyström should return None
    assert compute_Lp(x, mock_cov_func, gp_type="sparse_nystroem") is None

    # Test with invalid GaussianProcessType
    with pytest.raises(ValueError):
        compute_Lp(x, mock_cov_func, gp_type="unknown_type")

    # Test without specifying gp_type (it should be inferred)
    Lp_inferred = compute_Lp(x, mock_cov_func)
    assert Lp_inferred is not None
    assert isinstance(Lp_inferred, jnp.ndarray)

    # Test with custom sigma and jitter
    Lp_custom = compute_Lp(x, mock_cov_func, sigma=0.1, jitter=0.001)
    assert Lp_custom.shape == (3, 3)
    assert isinstance(Lp_custom, jnp.ndarray)

    # Test with no landmarks for 'sparse_cholesky'
    Lp_no_landmarks = compute_Lp(x, mock_cov_func, gp_type="sparse_cholesky")
    assert Lp_no_landmarks.shape == (3, 3)
    assert isinstance(Lp_no_landmarks, jnp.ndarray)


def test_compute_L():
    x = jnp.array([[1, 2], [3, 4], [5, 6], [8, 8]])
    landmarks = jnp.array([[1, 2], [3, 4], [5, 6]])
    mock_cov_func = mellon.cov.ExpQuad(1.1)

    # Test FULL type with Lp=None
    L = compute_L(x, mock_cov_func, gp_type="full")
    assert L.shape == (4, 4)
    assert isinstance(L, jnp.ndarray)

    # Test FULL type with Lp as an array
    Lp = jnp.array([[0.5, 0.1], [0.1, 0.5]])
    with pytest.raises(ValueError):
        compute_L(x, mock_cov_func, gp_type="full", Lp=Lp)

    # Test FULL_NYSTROEM type
    L = compute_L(x, mock_cov_func, gp_type="full_nystroem", rank=2)
    assert L.shape == (4, 2)
    assert isinstance(L, jnp.ndarray)

    # Test SPARSE_CHOLESKY with landmarks and Lp=None
    L = compute_L(x, mock_cov_func, gp_type="sparse_cholesky", landmarks=landmarks)
    assert L.shape == (4, 3)
    assert isinstance(L, jnp.ndarray)

    # Test SPARSE_CHOLESKY with landmarks and Lp as an array
    Lp = jnp.array([[0.5, 0.1], [0.1, 0.5]])
    L = compute_L(
        x,
        mock_cov_func,
        gp_type="sparse_cholesky",
        landmarks=landmarks[:2, :],
        Lp=Lp,
    )
    assert L.shape == (4, 2)
    assert isinstance(L, jnp.ndarray)

    with pytest.raises(ValueError):
        wrong_shape_Lp = jnp.array([[0.5, 0.1, 0.2], [0.1, 0.5, 0.2]])
        compute_L(
            x,
            mock_cov_func,
            gp_type="sparse_cholesky",
            landmarks=landmarks,
            Lp=wrong_shape_Lp,
        )

    # Test SPARSE_NYSTROEM
    L = compute_L(
        x, mock_cov_func, gp_type="sparse_nystroem", landmarks=landmarks, rank=2
    )
    assert L.shape == (4, 2)
    assert isinstance(L, jnp.ndarray)

    # Test with unknown gp_type
    with pytest.raises(ValueError):
        compute_L(x, mock_cov_func, gp_type="unknown")

    # Test with custom rank, sigma, and jitter
    L = compute_L(x, mock_cov_func, rank=2, sigma=0.1, jitter=0.001)
    assert L.shape == (4, 2)
    assert isinstance(L, jnp.ndarray)


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


def test_compute_nn_distances():
    # Test with different shapes of input
    x = jnp.array([[1, 2], [2, 3], [3, 4]])
    expected_output = jnp.array([jnp.sqrt(2), jnp.sqrt(2), jnp.sqrt(2)])
    assert jnp.allclose(compute_nn_distances(x), expected_output)

    # Test with non-positive distances
    x = jnp.array([[1, 2], [1, 2], [1, 2]])
    expected_output = jnp.array([0, 0, 0])
    assert jnp.allclose(compute_nn_distances(x), expected_output)

    # Test with varying distances
    x = jnp.array([[1, 1], [2, 2], [4, 4], [5, 5]])
    expected_output = jnp.array([jnp.sqrt(2), jnp.sqrt(2), jnp.sqrt(2), jnp.sqrt(2)])
    assert jnp.allclose(compute_nn_distances(x), expected_output)

    # Test with one instance
    x = jnp.array([[1, 2]])
    with pytest.raises(ValueError):
        compute_nn_distances(x)

    # Test with empty array
    x = jnp.array([])
    with pytest.raises(ValueError):
        compute_nn_distances(x)


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


def test_compute_initial_value():
    n = 2
    d = 2
    iv = mellon.parameters.compute_initial_value(
        jnp.arange(n) + 1, 3, 1, jnp.ones((n, d))
    )
    assert iv.dtype.kind == "f", "The initial value should have floating point numbers."
    assert iv.shape == (d,), "The initial value should have the right dimensionality."
    assert jnp.isfinite(iv).all(), "The initial value should have finite entries."
