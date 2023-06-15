import pytest
import jax.numpy as jnp
from mellon.validation import (
    _validate_time_x,
    _validate_float_or_int,
    _validate_positive_float,
    _validate_float,
    _validate_array,
    _validate_bool,
    _validate_string,
    _validate_float_or_iterable_numerical,
    _validate_positive_int,
    _validate_cov_func_curry,
    _validate_cov_func,
    _validate_1d,
)
from mellon.cov import Covariance


def test_validate_float_or_int():
    # Test with integer input
    assert _validate_float_or_int(10, "param") == 10

    # Test with float input
    assert _validate_float_or_int(10.5, "param") == 10.5

    # Test with string input
    with pytest.raises(ValueError):
        _validate_float_or_int("string", "param")

    # Test with None input and optional=True
    assert _validate_float_or_int(None, "param", optional=True) is None

    # Test with None input and optional=False
    with pytest.raises(ValueError):
        _validate_float_or_int(None, "param", optional=False)


def test_validate_positive_float():
    # Test with positive float input
    assert _validate_positive_float(10.5, "param") == 10.5

    # Test with negative float input
    with pytest.raises(ValueError):
        _validate_positive_float(-10.5, "param")

    # Test with positive integer input
    assert _validate_positive_float(10, "param") == 10.0

    # Test with negative integer input
    with pytest.raises(ValueError):
        _validate_positive_float(-10, "param")

    # Test with string input
    with pytest.raises(ValueError):
        _validate_positive_float("string", "param")

    # Test with None input and optional=True
    assert _validate_positive_float(None, "param", optional=True) is None

    # Test with None input and optional=False
    with pytest.raises(ValueError):
        _validate_positive_float(None, "param", optional=False)


def test_validate_positive_int():
    # Test with positive integer input
    assert _validate_positive_int(10, "param") == 10

    # Test with negative integer input
    with pytest.raises(ValueError):
        _validate_positive_int(-10, "param")

    # Test with float input
    with pytest.raises(ValueError):
        _validate_positive_int(10.5, "param")

    # Test with None input and optional=True
    assert _validate_positive_int(None, "param", optional=True) is None

    # Test with None input and optional=False
    with pytest.raises(ValueError):
        _validate_positive_int(None, "param", optional=False)


def test_validate_array():
    # Test with array-like input
    array = jnp.array([1, 2, 3])
    validated_array = _validate_array(array, "param", ndim=1)
    assert jnp.array_equal(validated_array, array)

    # Test with non-array input
    with pytest.raises(TypeError):
        _validate_array(10, "param")

    # Test with None input and optional=True
    assert _validate_array(None, "param", optional=True) is None

    # Test with None input and optional=False
    with pytest.raises(TypeError):
        _validate_array(None, "param", optional=False)

    # Test with incorrect number of dimensions
    with pytest.raises(ValueError):
        _validate_array(array, "param", ndim=2)


def test_validate_bool():
    # Test with bool input
    assert _validate_bool(True, "param") is True

    # Test with non-bool input
    with pytest.raises(TypeError):
        _validate_bool(10, "param")


def test_validate_string():
    # Test with string input
    assert _validate_string("test", "param") == "test"

    # Test with non-string input
    with pytest.raises(TypeError):
        _validate_string(10, "param")

    # Test with invalid choice
    with pytest.raises(ValueError):
        _validate_string("invalid", "param", choices=["valid", "test"])


def test_validate_float_or_iterable_numerical():
    # Test with float input
    assert _validate_float_or_iterable_numerical(10.5, "param") == 10.5

    # Test with iterable input
    array = jnp.array([1, 2, 3])
    validated_array = _validate_float_or_iterable_numerical(array, "param")
    assert jnp.array_equal(validated_array, array)

    # Test with non-numeric iterable input
    with pytest.raises(ValueError):
        _validate_float_or_iterable_numerical(["invalid", "input"], "param")

    # Test with non-numeric, non-iterable input
    with pytest.raises(ValueError):
        _validate_float_or_iterable_numerical("invalid", "param")


def test_validate_time_x():
    # Test with only 'x' and no 'times' or 'n_features'
    x = jnp.array([[1, 2], [3, 4], [5, 6]])
    result = _validate_time_x(x)
    assert jnp.array_equal(result, x)

    # Test with 'x' and 'times'
    times = jnp.array([1, 2, 3])
    expected_result = jnp.array([[1, 2, 1], [3, 4, 2], [5, 6, 3]])
    result = _validate_time_x(x, times)
    assert jnp.array_equal(result, expected_result)

    # Test with 'x' and 'times' with shape (n_samples, 1)
    times = jnp.array([[1], [2], [3]])
    result = _validate_time_x(x, times)
    assert jnp.array_equal(result, expected_result)

    # Test with 'x' and 'times' but mismatched number of samples
    times = jnp.array([1, 2])
    with pytest.raises(ValueError):
        _validate_time_x(x, times)

    # Test with 'x' and 'times' but 'times' is not 1D or 2D with 1 column
    times = jnp.array([[1, 1], [2, 2], [3, 3]])
    with pytest.raises(ValueError):
        _validate_time_x(x, times)

    # Test with 'x', 'times', and 'n_features' correct
    times = jnp.array([1, 2, 3])
    result = _validate_time_x(x, times, n_features=3)
    assert jnp.array_equal(result, expected_result)

    # Test with 'x', 'times', and 'n_features' incorrect
    with pytest.raises(ValueError):
        _validate_time_x(x, times, n_features=2)

    # Test with 'x', no 'times', and 'n_features' incorrect
    with pytest.raises(ValueError):
        _validate_time_x(x, n_features=3)

    # Test with scalar 'times' and 'cast_scalar' set to True
    times = 1
    expected_result = jnp.array([[1, 2, 1], [3, 4, 1], [5, 6, 1]])
    result = _validate_time_x(x, times, cast_scalar=True)
    assert jnp.array_equal(result, expected_result)


class CustomCovariance(Covariance):
    def __init__(self):
        pass

    def k(self):
        pass


def test_validate_cov_func_curry():
    # Test with both parameters as None
    with pytest.raises(ValueError):
        _validate_cov_func_curry(None, None, "cov_func_curry")

    # Test with valid covariance function curry
    cov_func_curry = CustomCovariance
    result = _validate_cov_func_curry(cov_func_curry, None, "cov_func_curry")
    assert result == cov_func_curry

    # Test with invalid covariance function curry
    cov_func_curry = "Invalid"
    with pytest.raises(ValueError):
        _validate_cov_func_curry(cov_func_curry, None, "cov_func_curry")


def test_validate_cov_func():
    # Test with valid covariance function
    cov_func = CustomCovariance()
    result = _validate_cov_func(cov_func, "cov_func")
    assert result == cov_func

    # Test with invalid covariance function
    cov_func = "Invalid"
    with pytest.raises(ValueError):
        _validate_cov_func(cov_func, "cov_func")

    # Test with None as optional
    result = _validate_cov_func(None, "cov_func", True)
    assert result is None

    # Test with None as not optional
    with pytest.raises(ValueError):
        _validate_cov_func(None, "cov_func", False)


def test_validate_1d():
    # Test with valid 1D array
    arr = jnp.array([1.2, 2.3, 3.4])
    result = _validate_1d(arr)
    assert jnp.allclose(result, arr), "Arrays not equal"

    # Test with a scalar
    scalar = 2.3
    result = _validate_1d(scalar)
    assert jnp.allclose(result, jnp.array([scalar])), "Arrays not equal"

    # Test with 2D array
    arr = jnp.array([[1.2, 2.3, 3.4], [4.5, 5.6, 6.7]])
    with pytest.raises(ValueError):
        _validate_1d(arr)

    # Test with string, should raise an error due to dtype mismatch
    string = "invalid"
    with pytest.raises(ValueError):
        _validate_1d(string)


def test_validate_float():
    # Test with valid float
    result = _validate_float(1.5, "param1")
    assert result == 1.5

    # Test with valid int
    result = _validate_float(2, "param1")
    assert result == 2.0

    # Test with None and optional
    result = _validate_float(None, "param1", optional=True)
    assert result is None

    # Test with None and not optional
    with pytest.raises(ValueError):
        _validate_float(None, "param1")

    # Test with invalid type
    with pytest.raises(ValueError):
        _validate_float("not a float", "param1")

    # Test with invalid type (non-numeric)
    with pytest.raises(ValueError):
        _validate_float([1, 2, 3], "param1")
