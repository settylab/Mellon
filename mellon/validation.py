from collections.abc import Iterable

from jax.numpy import asarray, concatenate

from .base_cov import Covariance


def _validate_time_x(x, times=None):
    """
    Validates and concatenates 'x' and 'times' if 'times' is provided.

    Parameters
    ----------
    x : array-like
        The training instances for which the density function will be estimated.
        Shape must be (n_samples, n_features).

    times : array-like, optional
        An array encoding the time points associated with each cell/row in 'x'.
        Shape must be either (n_samples,) or (n_samples, 1).

    Returns
    -------
    array-like
        The concatenated array of 'x' and 'times' (if provided).
    """

    x = _validate_array(x, "x")
    times = _validate_array(times, "times", optional=True)

    # Validate 'x' shape
    if x.ndim != 2:
        raise ValueError("'x' must be a 2D array.")

    if times is not None:
        # Validate 'times' shape
        if times.ndim == 1:
            times = times.reshape(-1, 1)
        elif times.ndim != 2 or times.shape[1] != 1:
            raise ValueError("'times' must be a 1D array or a 2D array with 1 column.")

        # Check that 'x' and 'times' have the same number of samples
        if x.shape[0] != times.shape[0]:
            raise ValueError("'x' and 'times' must have the same number of samples.")

        # Concatenate 'x' and 'times'
        x = concatenate((x, times), axis=1)

    return x


def _validate_float_or_int(value, param_name, optional=False):
    if value is None and optional:
        return None

    if not isinstance(value, (float, int)):
        raise ValueError(f"'{param_name}' should be a positive integer or float number")
    return value


def _validate_positive_float(value, param_name, optional=False):
    if value is None and optional:
        return None

    if not isinstance(value, (float, int)) or value < 0:
        raise ValueError(f"'{param_name}' should be a positive float number")
    return float(value)


def _validate_float(value, param_name, optional=False):
    if value is None and optional:
        return None

    if not isinstance(value, (float, int)):
        raise ValueError(f"'{param_name}' should be a float number")
    return float(value)


def _validate_positive_int(value, param_name, optional=False):
    if optional and value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"'{param_name}' should be a positive integer number")
    return value


def _validate_array(iterable, name, optional=False):
    if iterable is None and optional:
        return None

    if not isinstance(iterable, Iterable):
        raise TypeError(f"{name} should be iterable, got {type(iterable)} instead.")

    try:
        return asarray(iterable, dtype=float)
    except Exception:
        raise ValueError(f"Could not convert {name} to a numeric array.")


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

    if cov_func_curry is not None:
        if not issubclass(cov_func_curry, Covariance):
            raise ValueError(f"'{param_name}' must be a subclass of mellon.Covariance")
    return cov_func_curry


def _validate_cov_func(cov_func, param_name, optional=False):
    if cov_func is None and optional:
        return None
    if not isinstance(cov_func, Covariance):
        raise ValueError(
            f"'{param_name}' must be an instance of a subclass of mellon.Covariance"
        )
    return cov_func
