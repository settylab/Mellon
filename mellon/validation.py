from collections.abc import Iterable
import logging

from jax.numpy import (
    asarray,
    concatenate,
    isscalar,
    full,
    ndarray,
    where,
    isnan,
    isinf,
    squeeze,
)
from jax.numpy import sum as arraysum
from jax.numpy import min as arraymin
from jax.numpy import all as arrayall
from jax.errors import ConcretizationTypeError

logger = logging.getLogger(__name__)


def validate_time_x(x, times=None, n_features=None, cast_scalar=False):
    """
    Validates and concatenates 'x' and 'times' if 'times' is provided.

    If 'times' is provided, the function reshapes it if necessary and checks for
    matching number of samples in 'x' and 'times', before concatenating.

    If 'n_features' is provided, the function checks if 'x' has the correct number
    of features.

    Parameters
    ----------
    x : array-like
        The training instances for which the density function will be estimated.
        Shape must be (n_samples, n_features).

    times : array-like, optional
        An array encoding the time points associated with each cell/row in 'x'.
        Shape must be either (n_samples,) or (n_samples, 1).

    n_features : int, optional
        The expected number of features in 'x' including 'times' if it is provided.

    cast_scalar : bool, optional
        If true and 'times' is a scalar, it will be cast to a 1D array with a length
        equal to the number of samples in 'x'.

    Returns
    -------
    array-like
        The concatenated array of 'x' and 'times' (if provided), or 'x' if 'times'
        is not provided.

    Raises
    ------
    ValueError
        If 'times' is not a 1D or 2D array with 1 column, or 'x' and 'times' don't
        have the same number of samples, or 'x' does not have the expected number
        of features.
    """

    x = validate_array(x, "x", ndim=2)
    if (
        cast_scalar
        and times is not None
        and (isscalar(times) or all(s == 1 for s in times.shape))
    ):
        times = full(x.shape[0], times)
    times = validate_array(times, "times", optional=True, ndim=(1, 2))

    if times is not None:
        # Validate 'times' shape
        if times.ndim == 1:
            times = times.reshape(-1, 1)
        elif times.ndim != 2 or times.shape[1] != 1:
            raise ValueError("'times' must be a 1D array or a 2D array with 1 column.")

        # Check that 'x' and 'times' have the same number of samples
        if x.shape[0] != times.shape[0]:
            raise ValueError(
                "'x' and 'times' must have the same number of samples. "
                f"Got {x.shape[0]} for 'x' and {times.shape[0]} for 'times'."
            )

        # Concatenate 'x' and 'times'
        x = concatenate((x, times), axis=1)

    if n_features is not None:
        if x.shape[1] == n_features - 1 and times is None:
            raise ValueError(
                f"Expected {n_features} features including 'times' in 'x' but "
                f"only found {x.shape[1]} features and 'times' is not provided."
            )
        elif x.shape[1] != n_features:
            raise ValueError(
                f"Wrong number of features in 'x'. Expected {n_features} but got {x.shape[1]}."
            )

    return x


def validate_float_or_int(value, param_name, optional=False):
    """
    Validates whether a given value is a float or an integer, and not nan.

    Parameters
    ----------
    value : float, int, or string, or None
        The value to be validated. It should be a float, integer, or convertible to a float.
    param_name : str
        The name of the parameter to be used in the error message.
    optional : bool, optional
        Whether the value is optional. If optional and value is None, returns None. Default is False.

    Returns
    -------
    float or int
        The validated value as float or int.

    Raises
    ------
    ValueError
        If the value is not float, int, convertible to float, and not None when not optional.
    """

    if value is None and optional:
        return None

    if not isinstance(value, (float, int)):
        try:
            value = float(value)
        except TypeError:
            its_type = type(value)
            raise ValueError(
                f"'{param_name}' should be a positive integer or float number but is {its_type}"
            )

    if isnan(value):
        raise ValueError(f"'{param_name}' should be a non-NaN float number")
    return value


def validate_positive_float(value, param_name, optional=False):
    """
    Validates whether a given value is a positive float, and non-NaN.

    Parameters
    ----------
    value : float, int, or string, or None
        The value to be validated. It should be a positive float or convertible to a positive float.
    param_name : str
        The name of the parameter to be used in the error message.
    optional : bool, optional
        Whether the value is optional. If optional and value is None, returns None. Default is False.

    Returns
    -------
    float
        The validated value as a positive float.

    Raises
    ------
    ValueError
        If the value is not a positive float, not convertible to a positive float, NaN,
        and not None when not optional.
    """

    if value is None and optional:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        its_type = type(value)
        raise ValueError(f"'{param_name}' should be a float number but is {its_type}")

    if value <= 0:
        raise ValueError(f"'{param_name}' should be a positive float number")

    if isnan(value):
        raise ValueError(f"'{param_name}' should be a non-NaN float number")

    return value


def validate_float(value, param_name, optional=False):
    """
    Validates if the input is a float or can be converted to a float.

    Parameters
    ----------
    value : object
        Input to be checked.
    param_name : str
        Name of the input parameter, used for error messaging.
    optional : bool, optional
        If True and the input is None, None will be returned. Otherwise, if the input is None,
        it raises an error. By default, False.

    Returns
    -------
    float
        The input converted to float.

    Raises
    ------
    ValueError
        If the input is not a float, integer, or a one-element array, or if the input is None and
        optional=False.
    """
    if value is None:
        if optional:
            return None
        else:
            raise ValueError(
                f"'{param_name}' is None, but is required to be a float number"
            )

    if isinstance(value, ndarray) and value.size == 1:
        value = squeeze(value)

    if not isinstance(value, (float, int)):
        try:
            value = float(value)
        except TypeError:
            its_type = type(value)
            raise ValueError(
                f"'{param_name}' should be a float number but is {its_type}"
            )

    if isnan(value):
        raise ValueError(f"'{param_name}' should be a non-NaN float number")
    return value


def validate_positive_int(value, param_name, optional=False):
    """
    Validates whether a given value is a positive integer.

    Parameters
    ----------
    value : int or None
        The value to be validated. It should be a positive integer.
    param_name : str
        The name of the parameter to be used in the error message.
    optional : bool, optional
        Whether the value is optional. If optional and value is None, returns None. Default is False.

    Returns
    -------
    int or None
        The validated value as a positive integer, or None if the value is optional and None.

    Raises
    ------
    ValueError
        If the value is not a positive integer and not None when not optional.
    """

    if optional and value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"'{param_name}' should be a positive integer number")
    return value


def validate_array(iterable, name, optional=False, ndim=None):
    """
    Validates and converts an iterable to a numpy array of type float.
    Allows Jax's JVPTracer objects and avoids explicit conversion in these cases.

    Parameters
    ----------
    iterable : iterable or None
        The iterable to be validated and converted. If optional=True, can also be None.

    name : str
        The name of the variable, used in error messages.

    optional : bool, optional
        If True, 'iterable' can be None and the function will return None in this case.
        If False and 'iterable' is None, a TypeError is raised. Defaults to False.

    ndim : int or tuple of ints, optional
        The number of dimensions the array must have. If specified and the number of dimensions
        of 'iterable' is not in 'ndim', a ValueError is raised.

    Returns
    -------
    numpy.ndarray or jax._src.interpreters.ad.JVPTracer
        The input iterable converted to a numpy array of type float, or the input JVPTracer.

    Raises
    ------
    TypeError
        If 'iterable' is not iterable and not None, or if 'iterable' is None and optional=False.

    ValueError
        If 'iterable' can't be converted to a numpy array of type float,
        or if the number of dimensions of 'iterable' is not in 'ndim'.
    """

    if iterable is None:
        if optional:
            return None
        else:
            raise TypeError(f"'{name}' can't be None.")

    if hasattr(iterable, "todense"):
        array = asarray(iterable.todense(), dtype=float)
    elif isinstance(iterable, Iterable):
        array = asarray(iterable, dtype=float)
    else:
        raise TypeError(
            f"'{name}' should be iterable or sparse, got {type(iterable)} instead."
        )

    if ndim is not None:
        if isinstance(ndim, int):
            ndim = (ndim,)
        if array.ndim not in ndim:
            raise ValueError(
                f"'{name}' must be a {ndim}-dimensional array, got {array.ndim}-dimensional array instead."
            )

    return array


def validate_bool(value, name, optional=False):
    """
    Validates whether a given value is a boolean.

    Parameters
    ----------
    value : any
        The value to be validated.

    name : str
        The name of the parameter to be used in the error message.

    optional : bool, optional
        If True, 'value' can be None and the function will return None in this case.
        If False and 'value' is None, a TypeError is raised. Defaults to False.

    Returns
    -------
    bool
        The validated value as a boolean.

    Raises
    ------
    TypeError
        If the value is not of type bool.
    """

    if value is None:
        if optional:
            return None
        else:
            raise TypeError(f"'{name}' can't be None.")

    if not isinstance(value, bool):
        raise TypeError(f"{name} should be of type bool, got {type(value)} instead.")

    return value


def validate_string(value, name, choices=None):
    """
    Validates whether a given value is a string and optionally whether it is in a set of choices.

    Parameters
    ----------
    value : any
        The value to be validated.
    name : str
        The name of the parameter to be used in the error message.
    choices : list of str, optional
        A list of valid string choices. If provided, the value must be one of these choices.

    Returns
    -------
    str
        The validated value as a string.

    Raises
    ------
    TypeError
        If the value is not of type str.
    ValueError
        If the value is not one of the choices (if provided).
    """

    if not isinstance(value, str):
        raise TypeError(f"{name} should be of type str, got {type(value)} instead.")

    if choices and value not in choices:
        raise ValueError(f"{name} should be one of {choices}, got '{value}' instead.")

    return value


def validate_float_or_iterable_numerical(value, name, optional=False, positive=False):
    """
    Validates whether a given value is a float, integer, or iterable of numerical values,
    with an option to check for non-negativity.

    Parameters
    ----------
    value : int, float, Iterable or None
        The value to be validated.
    name : str
        The name of the parameter to be used in the error message.
    optional : bool, optional
        Whether the value is optional. If optional and value is None, returns None. Default is False.
    positive : bool, optional
        Whether to validate that the value is non-negative. Default is False.

    Returns
    -------
    float or ndarray
        The validated value as a float or a numeric array.

    Raises
    ------
    TypeError
        If the value is not of type int, float or iterable.
    ValueError
        If the value could not be converted to a numeric array (if iterable) or if the value is negative (if positive is True).
    """

    if value is None and optional:
        return None

    if isinstance(value, (int, float)):
        value = float(value)
        if positive and value < 0:
            raise ValueError(f"{name} should be a non-negative number or array")
        return value

    if isinstance(value, Iterable) and not isinstance(value, str):
        result = asarray(value, dtype=float)
        if positive and (result < 0).any():
            raise ValueError(f"All elements in {name} should be non-negative")
        return result

    raise TypeError(
        f"{name} should be of type int, float or iterable, got {type(value)} instead."
    )


def validate_1d(x):
    """
    Validates that `x` can be cast to a JAX array with exactly 1 dimension and float data type.

    Parameters
    ----------
    x : array-like or scalar
        The input data to be validated and cast.

    Returns
    -------
    array-like
        The validated and cast data. If `x` is a scalar, it is transformed into a 1D array with a single element.

    Raises
    ------
    ValueError
        If `x` cannot be cast to a JAX array with exactly 1 dimension.
    """
    x = asarray(x, dtype=float)

    # Add an extra dimension if x is a scalar
    if x.ndim == 0:
        x = x[None]

    if x.ndim != 1:
        raise ValueError("`x` must be exactly 1-dimensional.")

    return x


def validate_nn_distances(nn_distances, optional=False):
    """
    Validates and corrects nearest neighbor distances. Ensures all distances are
    positive and handles invalid values.

    Parameters
    ----------
    nn_distances : array-like or None
        The input nearest neighbor distances to be validated and corrected. If None
        and `optional` is True, the function returns None.

    optional : bool, optional
        If True, the function accepts `nn_distances` as None and returns None.

    Returns
    -------
    array-like or None
        The validated and corrected nearest neighbor distances. Identical or
        invalid cells have their distances set to the minimum positive distance
        found. Returns None if `nn_distances` is None and `optional` is True.

    Raises
    ------
    ValueError
        If all instances/cells are found to be identical or invalid.
    """
    if nn_distances is None and optional:
        return None
    elif nn_distances is None:
        message = "nn_distances are required but None is given."
        logger.error(message)
        raise ValueError(message)

    # Check for invalid values
    nan_mask = isnan(nn_distances)
    inf_mask = isinf(nn_distances)
    non_positive_mask = nn_distances <= 0
    nan_count = nan_mask.sum()
    inf_count = inf_mask.sum()
    negative_count = non_positive_mask.sum()
    total_invalid = nan_count + inf_count + negative_count

    bad_idx = nan_mask | inf_mask | non_positive_mask
    if arrayall(bad_idx):
        message = (
            f"All {total_invalid:,} computed nearest neighbor distances "
            "(`nn_distances` attribute) contain invalid values: "
            f"{nan_count:,} NaN, {inf_count:,} infinite, {negative_count:,} less or equal 0. "
            "Please check the input data. Setting invalid distances to the minimum positive value found."
        )
        logger.error(message)
        raise ValueError(message)

    min_positive = min(nn_distances[~bad_idx])
    nn_distances = where(~bad_idx, nn_distances, min_positive)

    if total_invalid > 0:
        logger.warning(
            f"The computed nearest neighbor distances (`nn_distances` attribute) contain "
            f"{total_invalid:,} invalid values: "
            f"{nan_count:,} NaN, {inf_count:,} infinite, {negative_count:,} less or equal 0. "
            "Please check the input data. Setting invalid distances to the minimum positive value found."
        )

    return nn_distances


def validate_k(k, n_samples):
    """Validate that k is an integer, at least 1, and strictly less than n_samples."""
    if not isinstance(k, int):
        message = f"Parameter k must be an integer, got {type(k).__name__} instead."
        logger.error(message)
        raise ValueError(message)
    if k < 1:
        message = f"Parameter k must be at least 1, got {k}."
        logger.error(message)
        raise ValueError(message)
    if k >= n_samples:
        message = (
            "Parameter k must be smaller than the number of samples. "
            f"Got k={k:,} with {n_samples:,} samples."
        )
        logger.error(message)
        raise ValueError(message)
