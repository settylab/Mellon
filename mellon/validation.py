from collections.abc import Iterable
import logging

from jax.numpy import asarray, concatenate, isscalar, full, ndarray
from jax.errors import ConcretizationTypeError

logger = logging.getLogger(__name__)


def _validate_params(rank, gp_type, n_samples, n_landmarks, landmarks):
    """
    Validates that rank, gp_type, n_samples, and n_landmarks are compatible.

    Parameters
    ----------
    rank : int or float or None
        The rank of the approximate covariance matrix. If `None`, it will
        be inferred based on the Gaussian Process type. If integer and greater
        than or equal to the number of samples or if float and
        equal to 1.0 or if 0, full rank is indicated. For FULL_NYSTROEM and
        SPARSE_NYSTROEM, it should be fractional 0 < rank < 1 or integer 0 < rank < n.
    gp_type : GaussianProcessType
        The type of the Gaussian Process. It helps to decide the rank.
    n_samples : int
        The number of samples/cells.
    n_landmarks : int
        Number of landmarks used in the approximation process.
    landmarks : array-like or None
        The given landmarks/inducing points.
    """

    n_landmarks = _validate_positive_int(n_landmarks, "n_landmarks")
    rank = _validate_float_or_int(rank, "rank")

    from .parameters import GaussianProcessType

    if not isinstance(gp_type, GaussianProcessType):
        message = (
            "gp_type needs to be a mellon.parameters.GaussianProcessType but is a "
            f"{type(gp_type)} instead."
        )
        logger.error(message)
        raise ValueError(message)

    # Validation logic for landmarks
    if landmarks is not None and n_landmarks != landmarks.shape[0]:
        n_spec = landmarks.shape[0]
        message = (
            f"There are {n_spec:,} landmarks specified but n_landmarks={n_landmarks:,}. "
            "Please omit specifying n_landmarks if landmarks are given."
        )
        logger.error(message)
        raise ValueError(message)

    if n_landmarks > n_samples:
        logger.warning(
            f"n_landmarks={n_landmarks:,} is larger than the number of cells {n_samples:,}."
        )

    # Validation logic for FULL and FULL_NYSTROEM types
    if (
        (
            gp_type == GaussianProcessType.FULL
            or gp_type == GaussianProcessType.FULL_NYSTROEM
        )
        and n_landmarks != 0
        and n_landmarks < n_samples
    ):
        message = (
            f"Gaussian Process type {gp_type} but n_landmarks={n_landmarks:,} is smaller "
            f"than the number of cells {n_samples:,}. Omit n_landmarks or set it to 0 to use "
            "a non-sparse Gaussian Process or omit gp_type to use a sparse one."
        )
        logger.error(message)
        raise ValueError(message)

    # Validation logic for SPARSE_CHOLESKY and SPARSE_NYSTROEM types
    elif (
        gp_type == GaussianProcessType.SPARSE_CHOLESKY
        or gp_type == GaussianProcessType.SPARSE_NYSTROEM
    ):
        if n_landmarks == 0:
            message = (
                f"Gaussian Process type {gp_type} but n_landmarks=0. Set n_landmarks "
                f"to a number smaller than the number of cells {n_samples:,} to use a"
                "sparse Gaussuian Process or omit gp_type to use a non-sparse one."
            )
            logger.error(message)
            raise ValueError(message)
        elif n_landmarks >= n_samples:
            message = (
                f"Gaussian Process type {gp_type} but n_landmarks={n_landmarks:,} is larger or "
                f"equal the number of cells {n_samples:,}. Reduce the number of landmarks to use a"
                "sparse Gaussuian Process or omit gp_type to use a non-sparse one.."
            )
            logger.error(message)
            raise ValueError(message)

    # Validation logic for rank
    if (
        type(rank) is int
        and (
            (gp_type == GaussianProcessType.SPARSE_CHOLESKY and rank >= n_landmarks)
            or (gp_type == GaussianProcessType.SPARSE_NYSTROEM and rank >= n_landmarks)
            or (gp_type == GaussianProcessType.FULL and rank >= n_samples)
            or (gp_type == GaussianProcessType.FULL_NYSTROEM and rank >= n_samples)
        )
        or type(rank) is float
        and rank >= 1.0
        or rank == 0
    ):
        # full rank is indicated
        if gp_type == GaussianProcessType.FULL_NYSTROEM:
            message = (
                f"Gaussian Process type {gp_type} requires "
                "fractional 0 < rank < 1 or integer "
                f"0 < rank < {n_samples:,} (number of cells) "
                f"but the actual rank is {rank}."
            )
            logger.error(message)
            raise ValueError(message)
        elif gp_type == GaussianProcessType.SPARSE_NYSTROEM:
            message = (
                f"Gaussian Process type {gp_type} requires "
                "fractional 0 < rank < 1 or integer "
                f"0 < rank < {n_landmarks:,} (number of landmakrs) "
                f"but the actual rank is {rank}."
            )
            logger.error(message)
            raise ValueError(message)
    elif (
        gp_type != GaussianProcessType.FULL_NYSTROEM
        and gp_type != GaussianProcessType.SPARSE_NYSTROEM
    ):
        message = (
            f"Given rank {rank} indicates Nyström rank reduction. "
            f"But the Gaussian Process type is set to {gp_type}."
        )
        logger.error(message)
        raise ValueError(message)


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
    """
    Validates whether a given value is a float or an integer.

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
    """
    Validates whether a given value is a positive float.

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
        If the value is not a positive float, not convertible to a positive float, and not None when not optional.
    """

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
    """
    Validates whether a given value is a boolean.

    Parameters
    ----------
    value : any
        The value to be validated.
    name : str
        The name of the parameter to be used in the error message.

    Returns
    -------
    bool
        The validated value as a boolean.

    Raises
    ------
    TypeError
        If the value is not of type bool.
    """

    if not isinstance(value, bool):
        raise TypeError(f"{name} should be of type bool, got {type(value)} instead.")

    return value


def _validate_string(value, name, choices=None):
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


def _validate_float_or_iterable_numerical(value, name, optional=False, positive=False):
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


def _validate_cov_func_curry(cov_func_curry, cov_func, param_name):
    """
    Validates covariance function curry type.

    Parameters
    ----------
    cov_func_curry : type or None
        The covariance function curry type to be validated.
    cov_func : mellon.Covariance or None
        An instance of covariance function.
    param_name : str
        The name of the parameter to be used in the error message.

    Returns
    -------
    type
        The validated covariance function curry type.

    Raises
    ------
    ValueError
        If both 'cov_func_curry' and 'cov_func' are None, or if 'cov_func_curry' is not a subclass of mellon.Covariance.
    """

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
    """
    Validates an instance of a covariance function.

    Parameters
    ----------
    cov_func : mellon.Covariance or None
        The covariance function instance to be validated.
    param_name : str
        The name of the parameter to be used in the error message.
    optional : bool, optional
        Whether the value is optional. If optional and value is None, returns None. Default is False.

    Returns
    -------
    mellon.Covariance or None
        The validated instance of a subclass of mellon.Covariance or None if optional.

    Raises
    ------
    ValueError
        If 'cov_func' is not an instance of a subclass of mellon.Covariance and not None when not optional.
    """

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
