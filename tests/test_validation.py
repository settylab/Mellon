import pytest
import jax.numpy as jnp
from mellon.validation import (
    validate_time_x,
    validate_float_or_int,
    validate_positive_float,
    validate_float,
    validate_array,
    validate_bool,
    validate_string,
    validate_float_or_iterable_numerical,
    validate_positive_int,
    validate_1d,
    validate_nn_distances,
)
from mellon.parameter_validation import (
    validate_params,
    validate_cov_func_curry,
    validate_cov_func,
)
from mellon.cov import Covariance
from mellon.util import GaussianProcessType


@pytest.mark.parametrize(
    "rank, gp_type, n_samples, n_landmarks, landmarks, exception_expected",
    [
        # Test valid cases
        (1.0, GaussianProcessType.FULL, 100, 0, None, None),
        (0.5, GaussianProcessType.FULL_NYSTROEM, 100, 100, None, None),
        (1.0, GaussianProcessType.SPARSE_CHOLESKY, 100, 50, None, None),
        (0.5, GaussianProcessType.SPARSE_NYSTROEM, 100, 50, jnp.zeros((50, 5)), None),
        # Test error for invalid rank
        (None, GaussianProcessType.FULL, 100, 0, None, ValueError),
        ("some_type", GaussianProcessType.FULL, 100, 50, None, ValueError),
        (0.9, GaussianProcessType.FULL, 100, 0, None, ValueError),
        # Test error for invalid gp_type (not a GaussianProcessType instance)
        (1.0, "some_type", 100, 0, None, ValueError),
        # Test error cases for landmarks
        (
            0.5,
            GaussianProcessType.SPARSE_NYSTROEM,
            100,
            51,
            jnp.zeros((50, 5)),
            ValueError,
        ),
        (None, GaussianProcessType.FULL, 100, 50, jnp.zeros((60, 5)), ValueError),
        (None, GaussianProcessType.FULL_NYSTROEM, 100, 50, None, ValueError),
        (0.5, GaussianProcessType.SPARSE_CHOLESKY, 100, 0, None, ValueError),
        (1.0, GaussianProcessType.FULL, 100, 10, None, ValueError),
        (0, GaussianProcessType.SPARSE_NYSTROEM, 100, 100, None, ValueError),
        (2.0, GaussianProcessType.FULL_NYSTROEM, 100, 0, None, ValueError),
        (100, GaussianProcessType.SPARSE_NYSTROEM, 100, 50, None, ValueError),
    ],
)
def testvalidate_params(
    rank, gp_type, n_samples, n_landmarks, landmarks, exception_expected
):
    if exception_expected:
        with pytest.raises(exception_expected):
            validate_params(rank, gp_type, n_samples, n_landmarks, landmarks)
    else:
        validate_params(rank, gp_type, n_samples, n_landmarks, landmarks)


def testvalidate_float_or_int():
    # Test with integer input
    assert validate_float_or_int(10, "param") == 10

    # Test with float input
    assert validate_float_or_int(10.5, "param") == 10.5

    # Test with string input
    with pytest.raises(ValueError):
        validate_float_or_int("string", "param")

    # Test with None input and optional=True
    assert validate_float_or_int(None, "param", optional=True) is None

    # Test with None input and optional=False
    with pytest.raises(ValueError):
        validate_float_or_int(None, "param", optional=False)

    # Test with nan value
    with pytest.raises(ValueError):
        validate_float_or_int(jnp.nan, "param")


def testvalidate_positive_float():
    # Test with positive float input
    assert validate_positive_float(10.5, "param") == 10.5

    # Test with negative float input
    with pytest.raises(ValueError):
        validate_positive_float(-10.5, "param")

    # Test with positive integer input
    assert validate_positive_float(10, "param") == 10.0

    # Test with negative integer input
    with pytest.raises(ValueError):
        validate_positive_float(-10, "param")

    # Test with string input
    with pytest.raises(ValueError):
        validate_positive_float("string", "param")

    # Test with None input and optional=True
    assert validate_positive_float(None, "param", optional=True) is None

    # Test with None input and optional=False
    with pytest.raises(ValueError):
        validate_positive_float(None, "param", optional=False)

    # Test with nan value
    with pytest.raises(ValueError):
        validate_positive_float(jnp.nan, "param")


def testvalidate_positive_int():
    # Test with positive integer input
    assert validate_positive_int(10, "param") == 10

    # Test with negative integer input
    with pytest.raises(ValueError):
        validate_positive_int(-10, "param")

    # Test with float input
    with pytest.raises(ValueError):
        validate_positive_int(10.5, "param")

    # Test with None input and optional=True
    assert validate_positive_int(None, "param", optional=True) is None

    # Test with None input and optional=False
    with pytest.raises(ValueError):
        validate_positive_int(None, "param", optional=False)

    # Test with nan value
    with pytest.raises(ValueError):
        validate_positive_int(jnp.nan, "param")


def testvalidate_array():
    # Test with array-like input
    array = jnp.array([1, 2, 3])
    validated_array = validate_array(array, "param", ndim=1)
    assert jnp.array_equal(validated_array, array)

    # Test with non-array input
    with pytest.raises(TypeError):
        validate_array(10, "param")

    # Test with None input and optional=True
    assert validate_array(None, "param", optional=True) is None

    # Test with None input and optional=False
    with pytest.raises(TypeError):
        validate_array(None, "param", optional=False)

    # Test with incorrect number of dimensions
    with pytest.raises(ValueError):
        validate_array(array, "param", ndim=2)


def testvalidate_bool():
    # Test with bool input
    assert validate_bool(True, "param") is True

    # Test with non-bool input
    with pytest.raises(TypeError):
        validate_bool(10, "param")


def testvalidate_string():
    # Test with string input
    assert validate_string("test", "param") == "test"

    # Test with non-string input
    with pytest.raises(TypeError):
        validate_string(10, "param")

    # Test with invalid choice
    with pytest.raises(ValueError):
        validate_string("invalid", "param", choices=["valid", "test"])


def testvalidate_float_or_iterable_numerical():
    # Test with positive numbers
    assert validate_float_or_iterable_numerical(5, "value") == 5.0
    assert jnp.allclose(
        validate_float_or_iterable_numerical([5, 6], "value"), jnp.asarray([5.0, 6.0])
    )

    # Test with negative numbers, without positive constraint
    assert validate_float_or_iterable_numerical(-5, "value") == -5.0
    assert jnp.allclose(
        validate_float_or_iterable_numerical([-5, -6], "value"),
        jnp.asarray([-5.0, -6.0]),
    )

    # Test with zero
    assert validate_float_or_iterable_numerical(0, "value") == 0.0

    # Test with positive=True
    assert validate_float_or_iterable_numerical(5, "value", positive=True) == 5.0

    # Test with negative numbers and positive=True
    with pytest.raises(ValueError):
        validate_float_or_iterable_numerical(-5, "value", positive=True)

    with pytest.raises(ValueError):
        validate_float_or_iterable_numerical([-5, 6], "value", positive=True)

    # Test with None and optional=True
    assert validate_float_or_iterable_numerical(None, "value", optional=True) is None

    # Test with None and optional=False
    with pytest.raises(TypeError):
        validate_float_or_iterable_numerical(None, "value", optional=False)

    # Test with non-numeric types
    with pytest.raises(TypeError):
        validate_float_or_iterable_numerical("string", "value")

    with pytest.raises(ValueError):
        validate_float_or_iterable_numerical(["string"], "value")

    # Test with mixed numeric and non-numeric iterable
    with pytest.raises(ValueError):
        validate_float_or_iterable_numerical([5, "string"], "value")


def testvalidate_time_x():
    # Test with only 'x' and no 'times' or 'n_features'
    x = jnp.array([[1, 2], [3, 4], [5, 6]])
    result = validate_time_x(x)
    assert jnp.array_equal(result, x)

    # Test with 'x' and 'times'
    times = jnp.array([1, 2, 3])
    expected_result = jnp.array([[1, 2, 1], [3, 4, 2], [5, 6, 3]])
    result = validate_time_x(x, times)
    assert jnp.array_equal(result, expected_result)

    # Test with 'x' and 'times' with shape (n_samples, 1)
    times = jnp.array([[1], [2], [3]])
    result = validate_time_x(x, times)
    assert jnp.array_equal(result, expected_result)

    # Test with 'x' and 'times' but mismatched number of samples
    times = jnp.array([1, 2])
    with pytest.raises(ValueError):
        validate_time_x(x, times)

    # Test with 'x' and 'times' but 'times' is not 1D or 2D with 1 column
    times = jnp.array([[1, 1], [2, 2], [3, 3]])
    with pytest.raises(ValueError):
        validate_time_x(x, times)

    # Test with 'x', 'times', and 'n_features' correct
    times = jnp.array([1, 2, 3])
    result = validate_time_x(x, times, n_features=3)
    assert jnp.array_equal(result, expected_result)

    # Test with 'x', 'times', and 'n_features' incorrect
    with pytest.raises(ValueError):
        validate_time_x(x, times, n_features=2)

    # Test with 'x', no 'times', and 'n_features' incorrect
    with pytest.raises(ValueError):
        validate_time_x(x, n_features=3)

    # Test with scalar 'times' and 'cast_scalar' set to True
    times = 1
    expected_result = jnp.array([[1, 2, 1], [3, 4, 1], [5, 6, 1]])
    result = validate_time_x(x, times, cast_scalar=True)
    assert jnp.array_equal(result, expected_result)


class CustomCovariance(Covariance):
    def __init__(self):
        pass

    def k(self):
        pass


def testvalidate_cov_func_curry():
    # Test with both parameters as None
    with pytest.raises(ValueError):
        validate_cov_func_curry(None, None, "cov_func_curry")

    # Test with valid covariance function curry
    cov_func_curry = CustomCovariance
    result = validate_cov_func_curry(cov_func_curry, None, "cov_func_curry")
    assert result == cov_func_curry

    # Test with invalid covariance function curry
    cov_func_curry = "Invalid"
    with pytest.raises(ValueError):
        validate_cov_func_curry(cov_func_curry, None, "cov_func_curry")


def testvalidate_cov_func():
    # Test with valid covariance function
    cov_func = CustomCovariance()
    result = validate_cov_func(cov_func, "cov_func")
    assert result == cov_func

    # Test with invalid covariance function
    cov_func = "Invalid"
    with pytest.raises(ValueError):
        validate_cov_func(cov_func, "cov_func")

    # Test with None as optional
    result = validate_cov_func(None, "cov_func", True)
    assert result is None

    # Test with None as not optional
    with pytest.raises(ValueError):
        validate_cov_func(None, "cov_func", False)


def testvalidate_1d():
    # Test with valid 1D array
    arr = jnp.array([1.2, 2.3, 3.4])
    result = validate_1d(arr)
    assert jnp.allclose(result, arr), "Arrays not equal"

    # Test with a scalar
    scalar = 2.3
    result = validate_1d(scalar)
    assert jnp.allclose(result, jnp.array([scalar])), "Arrays not equal"

    # Test with 2D array
    arr = jnp.array([[1.2, 2.3, 3.4], [4.5, 5.6, 6.7]])
    with pytest.raises(ValueError):
        validate_1d(arr)

    # Test with string, should raise an error due to dtype mismatch
    string = "invalid"
    with pytest.raises(ValueError):
        validate_1d(string)


def testvalidate_float():
    # Test with valid float
    result = validate_float(1.5, "param1")
    assert result == 1.5

    # Test with valid int
    result = validate_float(2, "param1")
    assert result == 2.0

    # Test with 1x1 array
    result = validate_float(jnp.array([2.0]), "param1")
    assert result == 2.0

    # Test with None and optional
    result = validate_float(None, "param1", optional=True)
    assert result is None

    # Test with None and not optional
    with pytest.raises(ValueError):
        validate_float(None, "param1")

    # Test with invalid type
    with pytest.raises(ValueError):
        validate_float("not a float", "param1")

    # Test with invalid type (non-numeric)
    with pytest.raises(ValueError):
        validate_float([1, 2, 3], "param1")

    # Test with nan value
    with pytest.raises(ValueError):
        validate_float(jnp.nan, "param1")


def testvalidate_nn_distances():
    # Test with all valid distances
    nn_distances = jnp.array([0.1, 0.5, 1.2, 0.3])
    result = validate_nn_distances(nn_distances)
    assert jnp.all(result == nn_distances), "Valid distances should not be changed."

    # Test with NaN values
    nn_distances = jnp.array([0.1, jnp.nan, 1.2, 0.3])
    result = validate_nn_distances(nn_distances)
    assert jnp.any(result != nn_distances), "NaN values should be replaced."
    assert not jnp.isnan(result).any(), "Result should not contain NaN values."

    # Test with infinite values
    nn_distances = jnp.array([0.1, jnp.inf, 1.2, 0.3])
    result = validate_nn_distances(nn_distances)
    assert jnp.any(result != nn_distances), "Infinite values should be replaced."
    assert not jnp.isinf(result).any(), "Result should not contain infinite values."

    # Test with negative values
    nn_distances = jnp.array([0.1, -0.5, 1.2, 0.3])
    result = validate_nn_distances(nn_distances)
    assert jnp.any(result != nn_distances), "Negative values should be replaced."
    assert jnp.all(result > 0), "Result should not contain negative values."

    # Test with a mix of invalid values
    nn_distances = jnp.array([0.1, jnp.nan, jnp.inf, -0.5, 1.2])
    result = validate_nn_distances(nn_distances)
    assert jnp.any(result != nn_distances), "Invalid values should be replaced."
    assert not jnp.isnan(result).any(), "Result should not contain NaN values."
    assert not jnp.isinf(result).any(), "Result should not contain infinite values."
    assert jnp.all(result > 0), "Result should not contain negative values."

    # Test with all invalid values
    nn_distances = jnp.array([jnp.nan, jnp.inf, -0.5])
    with pytest.raises(ValueError):
        validate_nn_distances(nn_distances)

    # Test with optional=True and nn_distances=None
    assert (
        validate_nn_distances(None, optional=True) is None
    ), "Should return None if optional is True and nn_distances is None."

    # Test with optional=False and nn_distances=None
    with pytest.raises(ValueError):
        validate_nn_distances(None, optional=False)
