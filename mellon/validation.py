from collections.abc import Iterable

from jax.numpy import asarray, concatenate, isscalar, full, ndarray
from jax.errors import ConcretizationTypeError


def _validate_time_x(x, times=None, n_features=None, cast_scalar=False):
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

    x = _validate_array(x, "x", ndim=2)
    if (
        cast_scalar
        and times is not None
        and (isscalar(times) or all(s == 1 for s in times.shape))
    ):
        times = full(x.shape[0], times)
    times = _validate_array(times, "times", optional=True, ndim=(1, 2))

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


def _validate_float_or_int(value, param_name, optional=False):
    if value is None and optional:
        return None

    if isinstance(value, (float, int)):
        return value
    try:
        return float(value)
    except TypeError:
        its_type = type(value)
        raise ValueError(
            f"'{param_name}' should be a positive integer or float number but is {its_type}"
        )


def _validate_positive_float(value, param_name, optional=False):
    if value is None and optional:
        return None

    try:
        value = float(value)
    except TypeError:
        its_type = type(value)
        raise ValueError(f"'{param_name}' should be a float number but is {its_type}")
    if value < 0:
        raise ValueError(f"'{param_name}' should be a positive float number")
    return value


def _validate_float(value, param_name, optional=False):
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
        try:
            value = value.item()
        except ConcretizationTypeError:
            # this must be a JAX tracer
            return value

    if isinstance(value, (float, int)):
        return value
    try:
        return float(value)
    except TypeError:
        its_type = type(value)
        raise ValueError(f"'{param_name}' should be a float number but is {its_type}")


def _validate_positive_int(value, param_name, optional=False):
    if optional and value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"'{param_name}' should be a positive integer number")
    return value


def _validate_array(iterable, name, optional=False, ndim=None):
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


def _validate_bool(value, name):
    if not isinstance(value, bool):
        raise TypeError(f"{name} should be of type bool, got {type(value)} instead.")

    return value


def _validate_string(value, name, choices=None):
    if not isinstance(value, str):
        raise TypeError(f"{name} should be of type str, got {type(value)} instead.")

    if choices and value not in choices:
        raise ValueError(f"{name} should be one of {choices}, got '{value}' instead.")

    return value


def _validate_float_or_iterable_numerical(value, name, optional=False):
    if value is None and optional:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, Iterable):
        try:
            return asarray(value, dtype=float)
        except Exception:
            raise ValueError(f"Could not convert {name} to a numeric array.")

    raise TypeError(
        f"{name} should be of type int, float or iterable, got {type(value)} instead."
    )


def _validate_cov_func_curry(cov_func_curry, cov_func, param_name):
    if cov_func_curry is None and cov_func is None:
        raise ValueError(
            "At least one of 'cov_func_curry' and 'cov_func' must not be None"
        )

    from .base_cov import Covariance

    if cov_func_curry is not None:
        if not isinstance(cov_func_curry, type) or not issubclass(
            cov_func_curry, Covariance
        ):
            raise ValueError(f"'{param_name}' must be a subclass of mellon.Covariance")
    return cov_func_curry


def _validate_cov_func(cov_func, param_name, optional=False):
    if cov_func is None and optional:
        return None
    from .base_cov import Covariance

    if not isinstance(cov_func, Covariance):
        raise ValueError(
            f"'{param_name}' must be an instance of a subclass of mellon.Covariance"
        )
    return cov_func


def _validate_1d(x):
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
