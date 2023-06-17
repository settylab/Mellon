from jax.numpy import exp, log, quantile, stack, unique, empty
from jax.numpy import sum as arraysum
from jax import random
from sklearn.cluster import k_means
from sklearn.linear_model import Ridge
from sklearn.neighbors import BallTree, KDTree
from .util import mle, local_dimensionality, Log, ensure_2d, DEFAULT_JITTER
from .decomposition import (
    _check_method,
    _full_rank,
    _full_decomposition_low_rank,
    _standard_low_rank,
    _modified_low_rank,
    DEFAULT_RANK,
    DEFAULT_METHOD,
)
from .validation import _validate_time_x, _validate_positive_float


DEFAULT_N_LANDMARKS = 5000

logger = Log()


def compute_landmarks(x, n_landmarks=DEFAULT_N_LANDMARKS):
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
    n_landmarks : int, optional
        The desired number of landmark points. If less than 2 or greater
        than the number of data points, the function will return None.
        Defaults to DEFAULT_N_LANDMARKS.

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
        return None
    logger.info(f"Computing {n_landmarks:,} landmarks with k-means clustering.")
    return k_means(x, n_landmarks, n_init=1)[0]


def compute_landmarks_rescale_time(
    x, ls, ls_time, times=None, n_landmarks=DEFAULT_N_LANDMARKS
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

    ls = _validate_positive_float(ls, "ls")
    ls_time = _validate_positive_float(ls_time, "ls_time")
    x = _validate_time_x(x, times)
    time_factor = ls / ls_time
    x = x.at[:, -1].set(x[:, -1] * time_factor)
    landmarks = compute_landmarks(x, n_landmarks=n_landmarks)
    if landmarks is not None:
        try:
            landmarks = landmarks.at[:, -1].set(landmarks[:, -1] / time_factor)
        except AttributeError:
            # landmarks is not a jax array
            landmarks[:, -1] = landmarks[:, -1] / time_factor
    return landmarks


def compute_distances(x, k):
    R"""
    Computes the distance to the k nearest neighbor for each training instance.

    :param x: The training instances.
    :type x: array-like
    :param k: The number of nearest neighbors to consider.
    :return: distances - The k observed nearest neighbor distances.
    :rtype: array-like
    """
    x = ensure_2d(x)
    if x.shape[1] >= 20:
        tree = BallTree(x, metric="euclidean")
    else:
        tree = KDTree(x, metric="euclidean")
    distances = tree.query(x, k=k + 1)[0][:, 1:]
    return distances


def compute_nn_distances(x):
    """
    Compute the distance to the nearest neighbor for each instance in the provided training dataset.

    This function calculates the Euclidean distance between each instance in the dataset and its closest neighbor.
    It returns an array of these distances, ordered in the same way as the input instances.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        An array-like object representing the training instances.

    Returns
    -------
    nn_distances : array-like of shape (n_samples,)
        An array of the Euclidean distances from each instance to its nearest neighbor in
        the input dataset. The ordering of the distances in this array corresponds to the
        ordering of the instances in the input data.

    """
    return compute_distances(x, 1)[:, 0]


def compute_nn_distances_within_time_points(x, times=None, normalize=False):
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

    normalize : bool, optional
        If True, distances are normalized by the number of samples within the same time point group.
        This normalization reduces potential bias in the density estimation arising from uneven
        sampling across different time points. Defaults to False.

    Returns
    -------
    nn_distances : array-like
        The observed nearest neighbor distances within the same time point group,
        preserving the order of instances in `x`.

    """
    x = _validate_time_x(x, times)
    unique_times = unique(x[:, -1])
    nn_distances = empty(x.shape[0])
    n_cells = x.shape[0]
    av_cells_per_tp = n_cells / len(unique_times)

    for time in unique_times:
        mask = x[:, -1] == time
        n_samples = arraysum(mask)
        if n_samples < 2:
            raise ValueError(
                f"Insufficient data: Only {n_samples} sample(s) found at time point {time}. "
                "Nearest neighbors cannot be computed with less than two samples per time point. "
                "Please confirm if you have provided the correct time axis. "
                "If the time points indeed have very few samples, consider aggregating nearby time points for better results, "
                "or you may specify `nn_distances` manually."
            )
        x_at_time = x[mask, :-1]
        nn_distances_at_time = compute_nn_distances(x_at_time)
        if normalize:
            nn_distances_at_time = nn_distances_at_time * n_samples / av_cells_per_tp
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
    R"""
    Computes the dimensionality of the data based on the average fractal
    dimension around n randomly selected cells.

    :param x: The training instances.
    :type x: array-like
    :param n: Number of samples.
    :type n: int
    :param seed: Random seed for sampling.
    :type seed: int
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
    return local_dims.mean()


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
    Computes ls equal to the geometric mean of the nearest neighbor distances times a constant.

    :param nn_distances: The observed nearest neighbor distances.
    :type nn_distances: array-like
    :return: ls - The geometric mean of the nearest neighbor distances times a constant.
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
        return cov_func_curry(ls, active_dims=slice(None, -1)) * cov_func_curry(
            ls_time, active_dims=-1
        )
    return cov_func_curry(ls)


def compute_L(
    x,
    cov_func,
    landmarks=None,
    rank=DEFAULT_RANK,
    method=DEFAULT_METHOD,
    jitter=DEFAULT_JITTER,
):
    R"""
    Compute an :math:`L` such that :math:`L L^\top \approx K`, where
    :math:`K` is the covariance matrix.

    :param x: The training instances.
    :type x: array-like
    :param cov_func: The Gaussian process covariance function.
    :type cov_func: function
    :param landmarks: The landmark points. If None, computes a full rank decompostion.
        Defaults to None.
    :type landmarks: array-like
    :param rank: The rank of the approximate covariance matrix.
        If rank is an int, an :math:`n \times` rank matrix
        :math:`L` is computed such that :math:`L L^\top \approx K`, the exact
        :math:`n \times n` covariance matrix.
        If rank is a float 0.0 :math:`\le` rank :math:`\le` 1.0, the rank/size
        of :math:`L` is selected such that the included eigenvalues of the covariance
        between landmark points account for the specified percentage of the
        sum of eigenvalues. Defaults to 0.999.
    :type rank: int or float
    :param method: Explicitly specifies whether rank is to be interpreted as a
        fixed number of eigenvectors or a percent of eigenvalues to include
        in the low rank approximation. Supports 'fixed', 'percent', or 'auto'.
        If 'auto', interprets rank as a fixed number of eigenvectors if it is
        an int and interprets rank as a percent of eigenvalues if it is a float.
        Defaults to 'auto'.
    :type method: str
    :param jitter: A small amount to add to the diagonal. Defaults to 1e-6.
    :type jitter: float
    :return: :math:`L` - A matrix such that :math:`L L^\top \approx K`.
    :rtype: array-like
    """
    x = ensure_2d(x)
    n_samples = x.shape[0]
    if landmarks is None:
        n = x.shape[0]
        method = _check_method(rank, n, method)

        if type(rank) is int and rank == n or type(rank) is float and rank == 1.0:
            logger.info(
                f"Doing full-rank Cholesky decomposition for {n_samples:,} samples."
            )
            return _full_rank(x, cov_func, jitter=jitter)
        else:
            logger.info(
                f"Doing full-rank singular value decomposition for {n_samples:,} samples."
            )
            return _full_decomposition_low_rank(
                x, cov_func, rank=rank, method=method, jitter=jitter
            )
    else:
        landmarks = ensure_2d(landmarks)

        n_landmarks = landmarks.shape[0]
        method = _check_method(rank, n_landmarks, method)

        if (
            type(rank) is int
            and rank == n_landmarks
            or type(rank) is float
            and rank == 1.0
        ):
            logger.info(
                "Doing low-rank Cholesky decomposition for "
                f"{n_samples:,} samples and {n_landmarks:,} landmarks."
            )
            return _standard_low_rank(x, cov_func, landmarks, jitter=jitter)
        else:
            logger.info(
                "Doing low-rank improved Nyström decomposition for "
                f"{n_samples:,} samples and {n_landmarks:,} landmarks."
            )
            return _modified_low_rank(
                x, cov_func, landmarks, rank=rank, method=method, jitter=jitter
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
    initial_dims = Ridge(fit_intercept=False).fit(L, target).coef_
    initial_dens = compute_initial_value(nn_distances, d, mu_dens, L)
    initial_value = stack([initial_dims, initial_dens])
    return initial_value
