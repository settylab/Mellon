import logging
from jax.numpy import (
    exp,
    log,
    quantile,
    stack,
    unique,
    empty,
    where,
    ndim,
    asarray,
    ndarray,
    ones,
    zeros,
    full,
)
from jax.numpy import sum as arraysum
from jax.numpy import any as arrayany
from jax import random
from sklearn.cluster import k_means
from sklearn.linear_model import Ridge
import pynndescent
from .util import (
    mle,
    local_dimensionality,
    ensure_2d,
    DEFAULT_JITTER,
    GaussianProcessType,
)
from .decomposition import (
    _full_rank,
    _full_decomposition_low_rank,
    _standard_low_rank,
    _modified_low_rank,
    DEFAULT_RANK,
    DEFAULT_SIGMA,
)
from .validation import (
    validate_array,
    validate_time_x,
    validate_positive_float,
    validate_float_or_int,
    validate_positive_int,
    validate_float_or_iterable_numerical,
    validate_k,
)
from .parameter_validation import (
    validate_params,
    validate_normalize_parameter,
)


DEFAULT_N_LANDMARKS = 5000
DEFAULT_RANDOM_SEED = 42

logger = logging.getLogger("mellon")


def compute_initial_zeros(x, L):
    return zeros((x.shape[0], L.shape[1]))


def compute_initial_ones(x, L):
    return ones(x.shape[0])


def compute_time_derivatives(predictor, x, times=None):
    if hasattr(predictor, "time_derivative"):
        return predictor.time_derivative(x, times)
    else:
        return zeros(x.shape[0])


def compute_density_gradient(predictor, x, times=None):
    if hasattr(predictor, "time_derivative"):
        return predictor.gradient(x, times)
    else:
        return predictor.gradient(x)


def compute_density_diffusion(predictor, x, times=None):
    if hasattr(predictor, "time_derivative"):
        sign, log_det = predictor.hessian_log_determinant(x, times)
    else:
        sign, log_det = predictor.hessian_log_determinant(x)


def compute_rank(gp_type):
    """
    Compute the appropriate rank reduction based on the given Gaussian Process type.

    Parameters
    ----------
    gp_type : GaussianProcessType
        The type of the Gaussian Process. It helps to decide the rank.

    Returns
    -------
    computed_rank : float or int or None
        The computed rank value based on the `gp_type`, `rank`, and shape of `x`.

    Raises
    ------
    ValueError
        If the given rank and Gaussian Process type conflict with each other.
    """

    if gp_type is None:
        return 1.0
    elif (gp_type == GaussianProcessType.FULL_NYSTROEM) or (
        gp_type == GaussianProcessType.SPARSE_NYSTROEM
    ):
        return DEFAULT_RANK
    else:
        return 1.0


def compute_n_landmarks(gp_type, n_samples, landmarks):
    """
    Compute the number of landmarks based on the given Gaussian Process type and landmarks.

    Parameters
    ----------
    gp_type : GaussianProcessType
        The type of the Gaussian Process. It helps to decide the number of landmarks.
    n_samples : array-like
        The number of samples/cells.
    landmarks : array-like or None
        The given landmarks. If specified, its shape determines the number of landmarks,
        unless conflicting with `n_landmarks`.

    Returns
    -------
    computed_n_landmarks : int
        The computed number of landmarks based on the `gp_type`, `n_landmarks`, shape of `x`, and `landmarks`.

    Raises
    ------
    ValueError
        If the given number of landmarks, Gaussian Process type, and landmarks conflict with each other.

    """
    if landmarks is not None:
        return landmarks.shape[0]

    if gp_type is None or gp_type == GaussianProcessType.FIXED:
        n_landmarks = min(n_samples, DEFAULT_N_LANDMARKS)
    elif (
        gp_type == GaussianProcessType.FULL
        or gp_type == GaussianProcessType.FULL_NYSTROEM
    ):
        n_landmarks = n_samples
    elif (
        gp_type == GaussianProcessType.SPARSE_CHOLESKY
        or gp_type == GaussianProcessType.SPARSE_NYSTROEM
    ):
        if n_samples <= DEFAULT_N_LANDMARKS:
            message = (
                f"Gaussian Process type {gp_type} and default "
                f"number of landmarks {DEFAULT_N_LANDMARKS:,} < "
                f"number of cells {n_samples:,}. Reduce n_landmarks below "
                f"the number of cells to use {gp_type}."
            )
            logger.warning(message)
        n_landmarks = DEFAULT_N_LANDMARKS
    else:
        n_landmarks = min(n_samples, DEFAULT_N_LANDMARKS)
        logger.warning(
            f"Unknown Gaussian Process type {gp_type}, using default "
            f"n_landmarks={n_landmarks:,}."
        )
    return n_landmarks


def compute_gp_type(n_landmarks, rank, n_samples):
    """
    Determines the type of Gaussian Process based on the landmarks, rank, and
    number of samples.

    Parameters
    ----------
    landmarks : array-like or None
        The landmark points for sparse computation.
    rank : int or float
        The rank of the approximate covariance matrix.
    n_samples : array-like
        The number of samples/cells.

    Returns
    -------
    GaussianProcessType
        One of the Gaussian Process types defined in the `GaussianProcessType` Enum.
    """

    rank = validate_float_or_int(rank, "rank", optional=True)
    n_landmarks = validate_positive_int(n_landmarks, "n_landmarks")
    n_samples = validate_positive_int(n_samples, "n_samples")

    if n_landmarks == 0 or n_landmarks >= n_samples:
        # Full model
        if (
            rank is None
            or type(rank) is int
            and (rank >= n_samples)
            or type(rank) is float
            and rank >= 1.0
            or rank == 0
        ):
            logger.info(
                "Using non-sparse Gaussian Process since n_landmarks "
                f"({n_landmarks:,}) >= n_samples ({n_samples:,}) and rank = {rank}."
            )
            return GaussianProcessType.FULL
        else:
            logger.info(
                "Using full Gaussian Process with Nyström rank reduction since n_landmarks "
                f"({n_landmarks:,}) >= n_samples ({n_samples:,}) and rank = {rank}."
            )
            return GaussianProcessType.FULL_NYSTROEM
    else:
        # Sparse model
        if (
            rank is None
            or type(rank) is int
            and (rank >= n_landmarks)
            or type(rank) is float
            and rank >= 1.0
            or rank == 0
        ):
            logger.info(
                "Using sparse Gaussian Process since n_landmarks "
                f"({n_landmarks:,}) < n_samples ({n_samples:,}) and rank = {rank}."
            )
            return GaussianProcessType.SPARSE_CHOLESKY
        else:
            logger.info(
                "Using sparse Gaussian Process with improved Nyström rank reduction since n_landmarks "
                f"({n_landmarks:,}) >= n_samples ({n_samples:,}) and rank = {rank}."
            )
            return GaussianProcessType.SPARSE_NYSTROEM


def compute_landmarks(x, gp_type=None, n_landmarks=DEFAULT_N_LANDMARKS, random_state=DEFAULT_RANDOM_SEED):
    R"""
    Computes the landmark points as k-means centroids.

    Landmark points are used to approximate the underlying structure of the
    input space. If the number of landmarks (`n_landmarks`) is zero or
    exceeds the number of available data points, the function will return None.

    Parameters
    ----------
    x : array-like
        The input data for which landmarks should be computed.
        Shape must be (n_samples, n_features).
    gp_type : GaussianProcessType
        The type of the Gaussian Process. If gp_type is 'fixed' then x
        is passed through as landmarks if n_landmakrs>=n_samples. Defaults to None.
    n_landmarks : int, optional
        The desired number of landmark points. If less than 2 or greater
        than the number of data points, the function will return None.
        Defaults to DEFAULT_N_LANDMARKS.
    random_state : int, optional
        Random seed for the k-means algorithm to ensure reproducible landmark selection.
        Defaults to DEFAULT_RANDOM_SEED (42).

    Returns
    -------
    landmark_points : array-like or None
        The coordinates of the computed landmark points, represented as
        k-means centroids. If no landmarks are computed, the function
        returns None. Shape is (n_landmarks, n_features).

    """
    if n_landmarks == 0:
        return None
    n = x.shape[0]
    x = ensure_2d(x)
    assert n_landmarks > 1, "n_landmarks musst be larger 1 or euqual to 0"
    if n_landmarks >= n:
        if gp_type == GaussianProcessType.FIXED:
            message = (
                f"Gaussin process type is {gp_type} and n_landmarks={n_landmarks:,} "
                f"are requested while only {n:,} datapoints are available. "
                f"Using all datapoints for {n:,} landmarks instead."
            )
            logger.warning(message)
            return x
        return None
    logger.info(f"Computing {n_landmarks:,} landmarks with k-means clustering (random_state={random_state}).")
    return k_means(x, n_landmarks, n_init=1, random_state=random_state)[0]


def compute_landmarks_rescale_time(
    x, ls, ls_time, times=None, n_landmarks=DEFAULT_N_LANDMARKS, random_state=DEFAULT_RANDOM_SEED
):
    R"""
    Computes landmark points for time-rescaled input data using k-means centroids.

    This function first rescales the temporal dimension of the input data by
    a factor derived from the spatial and temporal length scales (`ls` and `ls_time`).
    It then computes landmark points from the rescaled data. The last dimension
    of the landmarks is re-scaled back to the original time scale before being returned.

    Parameters
    ----------
    x : array-like
        The input data for which landmarks should be computed. If 'times' is
        None, the last column of 'x' is interpreted as the times.
        Shape must be (n_samples, n_features).
    ls : float
        Length scale of the spatial covariance kernel. Must be positive.
    ls_time : float
        Length scale of the temporal covariance kernel. Must be positive.
    times : array-like, optional
        An array encoding the time points associated with each sample in 'x'.
        If provided, it overrides the last column of 'x' as the times.
        Shape must be either (n_samples,) or (n_samples, 1).
    n_landmarks : int, optional
        The desired number of landmark points. Defaults to DEFAULT_N_LANDMARKS.
    random_state : int, optional
        Random seed for the k-means algorithm to ensure reproducible landmark selection.
        Defaults to DEFAULT_RANDOM_SEED (42).

    Returns
    -------
    landmark_points : array-like or None
        The coordinates of the computed landmark points, represented as
        k-means centroids in the original space, including the re-scaled
        temporal dimension. If no landmarks are computed, the function
        returns None. Shape is (n_landmarks, n_features).

    """
    if n_landmarks == 0:
        return None

    ls = validate_positive_float(ls, "ls")
    ls_time = validate_positive_float(ls_time, "ls_time")
    x = validate_time_x(x, times)
    time_factor = ls / ls_time
    x = x.at[:, -1].set(x[:, -1] * time_factor)
    landmarks = compute_landmarks(x, n_landmarks=n_landmarks, random_state=random_state)
    if landmarks is not None:
        try:
            landmarks = landmarks.at[:, -1].set(landmarks[:, -1] / time_factor)
        except AttributeError:
            # landmarks is not a jax array
            landmarks[:, -1] = landmarks[:, -1] / time_factor
    return landmarks


def compute_distances(x, k, seed=DEFAULT_RANDOM_SEED):
    """
    Compute the distance to the k nearest neighbor for each training instance.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        The training instances.
    k : int
        The number of nearest neighbors to consider. Must be a positive integer and strictly less
        than the number of samples.
    seed : int, optional (default=42)
        The seed for random number generation used during index initialization.

    Returns
    -------
    distances : array-like, shape (n_samples, k)
        The distances to the k nearest neighbors for each training instance.
        Note that the nearest neighbor of a point is itself, so the first neighbor (distance 0)
        is discarded.

    Raises
    ------
    ValueError
        If `x` is empty, if `k` is not an integer, if `k` is less than 1, or if `k` is greater
        than or equal to the number of samples.

    Notes
    -----
    Internally, NNDescent computes k+1 neighbors because every instance is its own nearest neighbor.
    The returned result discards the self-distance (first column). The `seed` parameter controls
    the random state used to initialize the NNDescent index.
    """
    x = validate_array(x, "x")
    x = ensure_2d(x)
    n_samples = x.shape[0]

    if n_samples == 0:
        message = "Input data x is empty."
        logger.error(message)
        raise ValueError(message)

    validate_k(k, n_samples)

    # The nearest neighbor of a point is itself, so request k+1 neighbors.
    index = pynndescent.NNDescent(
        x, n_neighbors=k + 1,
        metric="euclidean",
        random_state=seed,
    )

    _, distances = index.neighbor_graph
    return distances[:, 1:]


def compute_nn_distances(x, seed=DEFAULT_RANDOM_SEED):
    """
    Compute the distance to the nearest neighbor for each instance in the provided training dataset.

    This function calculates the Euclidean distance between each instance in the dataset and its closest neighbor.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        An array-like object representing the training instances.
    seed : int, optional (default=42)
        The seed for random number generation used during the NNDescent index initialization.

    Returns
    -------
    nn_distances : array-like of shape (n_samples,)
        An array of the Euclidean distances from each instance to its nearest neighbor in
        the input dataset. The ordering of the distances in this array corresponds to the
        ordering of the instances in the input data.

    Raises
    ------
    ValueError
        If there is an error in computing the nearest neighbor distances.
    """
    # Forward the seed parameter to compute_distances.
    return compute_distances(x, 1, seed=seed)[:, 0]


def _get_target_cell_count(normalize, time, av_cells_per_tp, unique_times):
    if isinstance(normalize, bool):
        return av_cells_per_tp
    if isinstance(normalize, dict):
        return normalize[time.item()]
    return normalize[unique_times.tolist().index(time)]


def compute_nn_distances_within_time_points(x, times=None, d=None, normalize=False):
    R"""
    Computes the distance to the nearest neighbor for each training instance
    within the same time point group. It retains the original order of instances in `x`.

    Parameters
    ----------
    x : array-like
        The training instances.
        If 'times' is None, the last column of 'x' is interpreted as the times.
        Shape must be (n_samples, n_features).

    times : array-like, optional
        An array encoding the time points associated with each cell/row in 'x'.
        If provided, it overrides the last column of 'x' as the times.
        Shape must be either (n_samples,) or (n_samples, 1).

    d : int, array-like or None
        The intrinsic dimensionality of the data, i.e., the dimensionality of the embedded
        manifold. Only required for the normalization.
        Defaults to None.

    normalize : bool, list, array-like, or dict, optional
        Controls the normalization for varying cell counts across time points to adjust for sampling bias
        by modifying the nearest neighbor distances.

        - If True, normalizes to simulate a constant total cell count divided by the number of time points.

        - If False, the raw cell counts per time point is reflected in the nearest neighbor distances.

        - If a list or array-like, assumes total cell counts for time points, ordered from earliest to latest.

        - If a dict, maps each time point to its total cell count. Must cover all unique time points.

        Default is False.

    Returns
    -------
    nn_distances : array-like
        The observed nearest neighbor distances within the same time point group,
        preserving the order of instances in `x`.

    """
    x = validate_time_x(x, times)
    unique_times = unique(x[:, -1])
    nn_distances = empty(x.shape[0])
    n_cells = x.shape[0]
    av_cells_per_tp = n_cells / len(unique_times)

    validate_normalize_parameter(normalize, unique_times)

    if normalize is not False and normalize is not None:
        d = validate_float_or_iterable_numerical(d, "d", optional=False, positive=True)
        if ndim(d) > 0 and len(d) != x.shape[0]:
            ld = len(d)
            raise ValueError(
                f"If `d` (length={ld:,}) is a vector then it needs to have one value "
                f"per cell in x (x.shape[0]={n_cells:,})."
            )
        logger.info(
            "Normalizing nearest neighbor distances correcting sampling bias for "
            f"{len(unique_times):,} different time points."
        )

    for time in unique_times:
        mask = x[:, -1] == time
        n_samples = arraysum(mask)
        if n_samples < 2:
            raise ValueError(
                f"Insufficient data: Only {n_samples} sample(s) found at time point {time}. "
                "Nearest neighbors cannot be computed with less than two samples per time point. "
                "Please confirm if you have provided the correct time axis. "
                "If the time points indeed have very few samples, consider aggregating nearby "
                "time points for better results, or you may specify `nn_distances` manually."
            )
        x_at_time = x[mask, :-1]
        nn_distances_at_time = compute_nn_distances(x_at_time)
        if normalize is not False and normalize is not None:
            target_cell_count = _get_target_cell_count(
                normalize, time, av_cells_per_tp, unique_times
            )
            factor = (n_samples / target_cell_count) ** (
                1 / d if ndim(d) == 0 else 1 / d[mask]
            )
            nn_distances_at_time = factor * nn_distances_at_time
        nn_distances = nn_distances.at[mask].set(nn_distances_at_time)

    return nn_distances


def compute_d(x):
    R"""
    Computes the dimensionality of the data equal to the size of axis 1.
    :param x: The training instances.
    :type x: array-like
    """
    if len(x.shape) < 2:
        return 1
    return x.shape[1]


def compute_d_factal(x, k=10, n=500, seed=432):
    """
    Computes the dimensionality of the data based on the average fractal
    dimension around `n` randomly selected cells.

    Parameters
    ----------
    x : array-like
        The training instances. Shape must be (n_samples, n_features).
    k : int, optional
        Number of nearest neighbors to use in the algorithm.
        Defaults to 10.
    n : int, optional
        Number of samples to randomly select.
        Defaults to 500.
    seed : int, optional
        Random seed for sampling. Defaults to 432.

    Returns
    -------
    float
        The average fractal dimension of the data.

    Warnings
    --------
    If `k` is greater than the number of samples in `x`, a warning will
    be logged, and `k` will be set to the number of samples.

    """
    if len(x.shape) < 2:
        return 1
    if n < x.shape[0]:
        key = random.PRNGKey(seed)
        idx = random.choice(key, x.shape[0], shape=(n,), replace=False)
        x_query = x[idx, ...]
    else:
        x_query = x
    local_dims = local_dimensionality(x, k=k, x_query=x_query)
    return local_dims.mean().item()


def compute_mu(nn_distances, d):
    R"""
    Computes mu equal to the 1th percentile of :math:`mle(nn\text{_}distances, d) - 10`,
    where :math:`mle =
    \log(\text{gamma}(d/2 + 1)) - (d/2) \cdot \log(\pi) - d \cdot \log(nn\text{_}distances)`

    :param nn_distances: The observed nearest neighbor distances.
    :type nn_distances: array-like
    :type d: The local dimensionality of the data.
    :type d: int
    :return: mu - The 1th percentile of :math:`mle(nn\text{_}distances, d) - 10`.
    :rtype: float
    """
    return quantile(mle(nn_distances, d), 0.01).item() - 10


def compute_ls(nn_distances):
    R"""
    Computes a length scale (ls) equal to the geometric mean of the positive nearest neighbor distances
    times a constant.

    :param nn_distances: The observed nearest neighbor distances. Must be non-empty.
    :type nn_distances: array-like
    :return: ls - The geometric mean of the nearest neighbor distances (after adjustment) times a constant.
    :rtype: float
    """

    return exp(log(nn_distances).mean() + 3.0).item()


def compute_cov_func(cov_func_curry, ls, ls_time=None):
    R"""
    Computes the Gaussian process covariance function from its generator and length scales.

    Parameters
    ----------
    cov_func_curry : function or type
        The covariance function generator.

    ls : float
        The length scale of the covariance function.

    ls_time : float, optional
        The time-specific length scale of the covariance function.
        If provided, the returned covariance function will account for the time-specific
        dimension of the input data (last dimension assumed to be time). Defaults to None.

    Returns
    -------
    cov_func : mellon.Covariance instance
        The resulting Gaussian process covariance function. If `ls_time` is
        provided, the covariance function is a product of two covariance functions,
        one for the feature dimensions and one for the time dimension.
        Otherwise, it's a single covariance function considering only the feature dimensions.
    """
    if ls_time is not None:
        return cov_func_curry(ls=ls, active_dims=slice(None, -1)) * cov_func_curry(
            ls=ls_time, active_dims=-1
        )
    return cov_func_curry(ls=ls)


def compute_Lp(
    x,
    cov_func,
    gp_type=None,
    landmarks=None,
    sigma=DEFAULT_SIGMA,
    jitter=DEFAULT_JITTER,
):
    R"""
    Compute a matrix :math:`L_p` such that :math:`L_p L_p^\top = \Sigma_p`
    where :math:`\Sigma_p` is the full rank covariance matrix on the
    inducing points. Unless a full Nyström method or sparse Nyström method
    is used, in which case None is returned.

    Parameters
    ----------
    x : array-like
        The training instances.
    cov_func : function
        The Gaussian process covariance function.
    gp_type : str or GaussianProcessType
        The type of sparcification used for the Gaussian Process:
         - 'full' None-sparse Gaussian Process
         - 'sparse_cholesky' Sparse GP using landmarks/inducing points,
            typically employed to enable scalable GP models.
    landmarks : array-like
        The landmark points.
    sigma : float, optional
        Noise standard deviation of the data we condition on. Defaults to 0.
    jitter : float, optional
        A small amount to add to the diagonal. Defaults to 1e-6.

    Returns
    -------
    array-like or None
        :math:`L_p` - A matrix such that :math:`L_p L_p^\top = \Sigma_p`,
        or None if using full or sparse Nyström.
    """
    x = ensure_2d(x)
    n_samples = x.shape[0]
    if landmarks is None:
        n_landmarks = n_samples
        landmarks = x
    else:
        landmarks = ensure_2d(landmarks)
        n_landmarks = landmarks.shape[0]
    gp_type = GaussianProcessType.from_string(gp_type, optional=True)
    if gp_type is None:
        gp_type = compute_gp_type(n_landmarks, 1.0, n_samples)

    if (
        gp_type == GaussianProcessType.FULL_NYSTROEM
        or gp_type == GaussianProcessType.SPARSE_NYSTROEM
    ):
        return None
    elif gp_type == GaussianProcessType.FULL:
        logger.info("Computing Lp.")
        return _full_rank(x, cov_func, sigma=sigma, jitter=jitter)
    elif (
        gp_type == GaussianProcessType.SPARSE_CHOLESKY
        or gp_type == GaussianProcessType.FIXED
    ):
        return _full_rank(landmarks, cov_func, sigma=sigma, jitter=jitter)
    else:
        message = f"Unknown Gaussian Process type {gp_type}."
        logger.error(message)
        raise ValueError(message)


def validate_compute_L_input(x, cov_func, gp_type, landmarks, Lp, rank, sigma, jitter):
    """
    Validate input for the fuction compute_L.

    Returns
    -------
    x : array-like
        The training instances with at least 2 dimensions (n_samples, n_dims).
    n_landmarks : int
        The number of landmarks.
    n_samples : int
        The number of samples/cells.
    gp_type : mellon.util.GaussianProcessType
        The type of Gaussian Process to use.
    rank : float, optional
        The rank of the approximate covariance matrix.

    Raises
    ------
    ValueError
        If any of the inputs are inconsistent or violate constraints.
    """
    jitter = validate_positive_float(jitter, "jitter")
    rank = validate_float_or_int(rank, "rank", optional=True)

    n_samples = x.shape[0]
    if landmarks is None:
        n_landmarks = n_samples
    else:
        n_landmarks = landmarks.shape[0]
    gp_type = GaussianProcessType.from_string(gp_type, optional=True)
    if rank is None:
        rank = compute_rank(gp_type)
    if gp_type is None:
        gp_type = compute_gp_type(n_landmarks, rank, n_samples)
    validate_params(rank, gp_type, n_samples, n_landmarks, landmarks)

    if (
        gp_type == GaussianProcessType.FULL
        and Lp is not None
        and Lp.shape != (n_samples, n_samples)
    ):
        message = (
            f" Wrong shape of Lp {Lp.shape} for {gp_type} and {n_samples:,} samples."
        )
        logger.error(message)
        raise ValueError(message)
    elif (
        (
            gp_type == GaussianProcessType.SPARSE_CHOLESKY
            or gp_type == GaussianProcessType.FIXED
        )
        and Lp is not None
        and Lp.shape != (n_landmarks, n_landmarks)
    ):
        message = f" Wrong shape of Lp {Lp.shape} for {gp_type} and {n_landmarks:,} landmarks."
        logger.error(message)
        raise ValueError(message)

    x = ensure_2d(x)
    if landmarks is not None:
        landmarks = ensure_2d(landmarks)

    return x, landmarks, n_landmarks, n_samples, gp_type, rank


def compute_L(
    x,
    cov_func,
    gp_type=None,
    landmarks=None,
    Lp=None,
    rank=None,
    sigma=DEFAULT_SIGMA,
    jitter=DEFAULT_JITTER,
):
    R"""
    Compute a low rank :math:`L` such that :math:`L L^\top \approx K`,
    where :math:`K` is the full rank covariance matrix on `x`.

    Parameters
    ----------
    x : array-like
        The training instances.
    cov_func : function
        The Gaussian process covariance function.
    gp_type : str or GaussianProcessType
        The type of sparcification used for the Gaussian Process:
         - 'full' None-sparse Gaussian Process
         - 'full_nystroem' Sparse GP with Nyström rank reduction without landmarks,
            which lowers the computational complexity.
         - 'sparse_cholesky' Sparse GP using landmarks/inducing points,
            typically employed to enable scalable GP models.
         - 'sparse_nystroem' Sparse GP using landmarks or inducing points,
            along with an improved Nyström rank reduction method that balances
            accuracy with efficiency.
    landmarks : array-like, optional
        The landmark points. If None, computes a full rank decomposition. Defaults to None.
    rank : int or float, optional
        The rank of the approximate covariance matrix.
        If rank is an int, an :math:`n \times` rank matrix
        :math:`L` is computed such that :math:`L L^\top \approx K`, the exact
        :math:`n \times n` covariance matrix.
        If rank is a float 0.0 :math:`\le` rank :math:`\le` 1.0, the rank/size
        of :math:`L` is selected such that the included eigenvalues of the covariance
        between landmark points account for the specified percentage of the
        sum of eigenvalues. Defaults to 0.99 if gp_type indicates Nyström.
    sigma : float, array-like, optional
        Noise standard deviation of the data we condition on. Defaults to 0.
    jitter : float, optional
        A small amount to add to the diagonal. Defaults to 1e-6.
    Lp : array-like, optional
        Prespecified matrix :math:`L_p` sich that :math:`L_p L_p^\top = \Sigma_p`
        where :math:`\Sigma_p` is the full rank covariance matrix on the
        inducing points. Defaults to None.

    Returns
    -------
    array-like
        :math:`L` - Matrix such that :math:`L L^\top \approx K`.

    Raises
    ------
    ValueError
        If the Gaussian Process type is unknown or if the shape of Lp is incorrect.
    """
    x, landmarks, n_landmarks, n_samples, gp_type, rank = validate_compute_L_input(
        x, cov_func, gp_type, landmarks, Lp, rank, sigma, jitter
    )

    if gp_type == GaussianProcessType.FULL:
        if Lp is None:
            return _full_rank(x, cov_func, sigma=sigma, jitter=jitter)
        return Lp
    elif gp_type == GaussianProcessType.FULL_NYSTROEM:
        return _full_decomposition_low_rank(
            x, cov_func, rank=rank, sigma=sigma, jitter=jitter
        )
    elif (
        gp_type == GaussianProcessType.SPARSE_CHOLESKY
        or gp_type == GaussianProcessType.FIXED
    ):
        if Lp is None:
            return _standard_low_rank(
                x, cov_func, landmarks, sigma=sigma, jitter=jitter
            )
        return _standard_low_rank(
            x, cov_func, landmarks, Lp=Lp, sigma=sigma, jitter=jitter
        )
    elif gp_type == GaussianProcessType.SPARSE_NYSTROEM:
        return _modified_low_rank(
            x,
            cov_func,
            landmarks,
            rank=rank,
            sigma=sigma,
            jitter=jitter,
        )


def compute_initial_value(nn_distances, d, mu, L):
    R"""
    Computes the initial value for Maximum A Posteriori optimization with Ridge regression,
    such that the initial value :math:`z` minimizes
    :math:`||Lz + mu - mle(nn\text{_}distances, d)|| + ||z||`.

    :param nn_distances: The observed nearest neighbor distances.
    :type nn_distances: array-like
    :param d: The local dimensionality of the data.
    :type d: int
    :param mu: The Gaussian Process mean.
    :type mu: int
    :param L: A matrix such that :math:`L L^\top \approx K`, where :math:`K`
        is the covariance matrix.
    :type L: array-like
    :return: initial_value - The argmin :math:`z`.
    :rtype: array-like
    """
    target = mle(nn_distances, d) - mu
    return Ridge(fit_intercept=False).fit(L, target).coef_


def compute_initial_dimensionalities(x, mu_dim, mu_dens, L, nn_distances, d):
    R"""
    Computes an initial guess for the log dimensionality and log density at every cell state
    with Ridge regression.

    :param x: The cell states.
    :type x: array-like
    :param mu: The Gaussian Process mean.
    :type mu: int
    :param L: A matrix such that :math:`L L^\top \approx K`, where :math:`K`
        is the covariance matrix.
    :type L: array-like
    :param nn_distances: The observed nearest neighbor distances.
    :type nn_distances: array-like
    :param d: The local dimensionality of the data.
    :type d: array-like
    :return: initial_value
    :rtype: array-like
    """
    target = log(d) - mu_dim
    if asarray(target).size == 1:
        target = full(L.shape[0], target)
    initial_dims = Ridge(fit_intercept=False).fit(L, target).coef_
    initial_dens = compute_initial_value(nn_distances, d, mu_dens, L)
    initial_value = stack([initial_dims, initial_dens])
    return initial_value


def compute_average_cell_count(x, normalize):
    """
    Compute the average cell count based on the `normalize` parameter and the input data `x`.

    Parameters
    ----------
    x : jax.numpy.ndarray
        Input array with shape (n_samples, n_features).
        The last column is assumed to contain the time identifiers.

    normalize : bool, list, jax.numpy.ndarray, dict, or None
        The parameter controlling the normalization.

        - If True or None, returns the average cell count computed from `x`.

        - If a list or jax.numpy.ndarray, returns the average of the list or array.

        - If a dict, returns the average of the dict values.

    Returns
    -------
    float
        The average cell count computed based on the `normalize` parameter and `x`.

    Raises
    ------
    ValueError
        If the type of `normalize` is not recognized.
    """
    n_cells = x.shape[0]
    unique_times = unique(x[:, -1])
    n_unique_times = unique_times.shape[0]

    if normalize is None or isinstance(normalize, bool):
        return n_cells / n_unique_times

    if isinstance(normalize, dict):
        return sum(normalize.values()) / n_unique_times

    if isinstance(normalize, (list, ndarray)):
        return arraysum(asarray(normalize)) / len(normalize)

    raise ValueError(f"Unrecognized type for 'normalize': {type(normalize)}")
