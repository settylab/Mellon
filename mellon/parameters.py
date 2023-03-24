from jax.numpy import exp, log, quantile
from sklearn.cluster import k_means
from sklearn.linear_model import Ridge
from sklearn.neighbors import BallTree, KDTree
from .util import mle, DEFAULT_JITTER
from .decomposition import (
    _check_method,
    _full_rank,
    _full_decomposition_low_rank,
    _standard_low_rank,
    _modified_low_rank,
    DEFAULT_RANK,
    DEFAULT_METHOD,
)


DEFAULT_N_LANDMARKS = 5000


def compute_landmarks(x, n_landmarks=DEFAULT_N_LANDMARKS):
    R"""
    Computes the landmark points as k-means centroids. If n_landmarks is less
    than 1 or greater than or equal to the number of training instances, returns None.

    :param x: The training instances.
    :type x: array-like
    :param n_landmarks: The number of landmark points.
    :type n_landmarks: int
    :return: landmark_points - k-means centroids.
    :rtype: array-like
    """
    n = x.shape[0]
    if len(x.shape) < 2:
        x = x[:, None]
    assert n_landmarks > 1, "n_landmarks musst be larger 1"
    if n_landmarks >= n:
        return x
    return k_means(x, n_landmarks, n_init=1)[0]


def compute_nn_distances(x):
    R"""
    Computes the distance to the nearest neighbor for each training instance.

    :param x: The training instances.
    :type x: array-like
    :return: nn_distances - The observed nearest neighbor distances.
    :rtype: array-like
    """
    if len(x.shape) < 2:
        x = x[:, None]
    if x.shape[1] >= 20:
        tree = BallTree(x, metric="euclidean")
    else:
        tree = KDTree(x, metric="euclidean")
    nn = tree.query(x, k=2)[0][:, 1]
    return nn


def compute_d(x):
    R"""
    Computes the dimensionality of the data equal to the size of axis 1.

    :param x: The training instances.
    :type x: array-like
    """
    if len(x.shape) < 2:
        return 1
    return x.shape[1]


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
    return quantile(mle(nn_distances, d), 0.01) - 10


def compute_ls(nn_distances):
    R"""
    Computes ls equal to the geometric mean of the nearest neighbor distances times a constant.

    :param nn_distances: The observed nearest neighbor distances.
    :type nn_distances: array-like
    :return: ls - The geometric mean of the nearest neighbor distances times a constant.
    :rtype: float
    """
    return exp(log(nn_distances).mean() + 3.0)


def compute_cov_func(cov_func_curry, ls):
    R"""
    Computes the Gaussian process covariance function from its generator and length scale.

    :param cov_func_curry: The covariance function generator.
    :type cov_func_curry: function or type
    :param ls: The length scale of the covariance function.
    :type ls: float
    :return: cov_func - The Gaussian process covariance function k(x, y) :math:`\rightarrow` float.
    :rtype: function
    """
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
    if len(x.shape) < 2:
        x = x[:, None]
    if landmarks is None:
        n = x.shape[0]
        method = _check_method(rank, n, method)

        if type(rank) is int and rank == n or type(rank) is float and rank == 1.0:
            return _full_rank(x, cov_func, jitter=jitter)
        else:
            return _full_decomposition_low_rank(
                x, cov_func, rank=rank, method=method, jitter=jitter
            )
    else:
        if len(landmarks.shape) < 2:
            landmarks = landmarks[:, None]

        n_landmarks = landmarks.shape[0]
        method = _check_method(rank, n_landmarks, method)

        if (
            type(rank) is int
            and rank == n_landmarks
            or type(rank) is float
            and rank == 1.0
        ):
            return _standard_low_rank(x, cov_func, landmarks, jitter=jitter)
        else:
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
