import logging
from jax.numpy import ndarray
from .util import GaussianProcessType
from .base_cov import Covariance
from .validation import (
    validate_positive_int,
    validate_float_or_int,
)

logger = logging.getLogger("mellon")


def validate_landmark_params(n_landmarks, landmarks):
    """
    Validates that n_landmarks and landmarks are compatible.

    Parameters
    ----------
    n_landmarks : int
        Number of landmarks used in the approximation process.
    landmarks : array-like or None
        The given landmarks/inducing points.
    """
    if landmarks is not None and n_landmarks != landmarks.shape[0]:
        n_spec = landmarks.shape[0]
        message = (
            f"There are {n_spec:,} landmarks specified but n_landmarks={n_landmarks:,}. "
            "Please omit specifying n_landmarks if landmarks are given."
        )
        logger.error(message)
        raise ValueError(message)


def validate_rank_params(gp_type, n_samples, rank, n_landmarks):
    """
    Validates that rank, n_landmarks, and gp_type are compatible.

    Parameters
    ----------
    gp_type : GaussianProcessType
        The type of the Gaussian Process. It helps to decide the rank.
    n_samples : int
        The number of samples/cells.
    rank : int or float or None
        The rank of the approximate covariance matrix. If `None`, it will
        be inferred based on the Gaussian Process type. If integer and greater
        than or equal to the number of samples or if float and
        equal to 1.0 or if 0, full rank is indicated. For FULL_NYSTROEM and
        SPARSE_NYSTROEM, it should be fractional 0 < rank < 1 or integer 0 < rank < n.
    n_landmarks : int
        Number of landmarks used in the approximation process.
    """
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


def validate_gp_type(gp_type, n_samples, n_landmarks):
    """
    Validates that gp_type, n_samples, and n_landmarks are compatible.

    Parameters
    ----------
    gp_type : GaussianProcessType
        The type of the Gaussian Process. It helps to decide the rank.
    n_samples : int
        The number of samples/cells.
    n_landmarks : int
        Number of landmarks used in the approximation process.
    """
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
                "sparse Gaussuian Process or omit gp_type to use a non-sparse one."
            )
            logger.warning(message)
            raise ValueError(message)


def validate_params(rank, gp_type, n_samples, n_landmarks, landmarks):
    """
    Validates that rank, gp_type, n_samples, n_landmarks, and landmarks are compatible.

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

    n_landmarks = validate_positive_int(n_landmarks, "n_landmarks")
    rank = validate_float_or_int(rank, "rank")

    if not isinstance(gp_type, GaussianProcessType):
        message = (
            "gp_type needs to be a mellon.util.GaussianProcessType but is a "
            f"{type(gp_type)} instead."
        )
        logger.error(message)
        raise ValueError(message)

    # Validation logic for landmarks
    validate_landmark_params(n_landmarks, landmarks)
    if n_landmarks > n_samples and gp_type != GaussianProcessType.FIXED:
        logger.warning(
            f"n_landmarks={n_landmarks:,} is larger than the number of cells {n_samples:,}."
        )

    validate_gp_type(gp_type, n_samples, n_landmarks)

    # Validation logic for rank
    validate_rank_params(gp_type, n_samples, rank, n_landmarks)


def validate_cov_func_curry(cov_func_curry, cov_func, param_name):
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

    if cov_func_curry is not None:
        if not isinstance(cov_func_curry, type) or not issubclass(
            cov_func_curry, Covariance
        ):
            raise ValueError(f"'{param_name}' must be a subclass of mellon.Covariance")
    return cov_func_curry


def validate_cov_func(cov_func, param_name, optional=False):
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

    if not isinstance(cov_func, Covariance):
        raise ValueError(
            f"'{param_name}' must be an instance of a subclass of mellon.Covariance"
        )
    return cov_func


def validate_normalize_parameter(normalize, unique_times):
    """
    Used in parameters.compute_nn_distances_within_time_points to validate input.
    """
    if isinstance(normalize, dict):
        missing_times = [t for t in unique_times if t.item() not in normalize]
        if missing_times:
            raise ValueError(
                f"Missing time point(s) in normalization dictionary: {missing_times}"
            )
    elif isinstance(normalize, (list, ndarray)) and len(normalize) != len(unique_times):
        raise ValueError(
            "Length of the normalize list or array must match the number of unique time points."
        )
