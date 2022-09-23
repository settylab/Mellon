from jax.numpy import sqrt, ceil, exp, log, geomspace, quantile
from jax.numpy.linalg import norm
from sklearn.cluster import k_means
from sklearn.linear_model import Ridge
from sklearn.neighbors import BallTree, KDTree
from .util import mle


DEFAULT_N_LANDMARKS = 5000


def compute_landmarks(x, n_landmarks=DEFAULT_N_LANDMARKS):
    R"""
    Computes the landmark points as k-means centroids.
    
    :param x: Training instances.
    :type x: array-like
    :type n_landmarks: Number of landmark points. If less than 1 or greater than
        or equal to the number of training instances, returns None.
    :type n_landmarks: int
    :return: landmark_points - k-means centroids.
    :rtype: array-like
    """
    n = x.shape[0]
    if (n_landmarks < 1) or (n_landmarks >= n):
        return
    return k_means(x, n_landmarks, n_init=1)[0]


def compute_nn_distances(x):
    R"""
    Computes the distance to the nearest neighbor for each training instance.

    :param x: Training instances.
    :type x: array-like
    :return: nn_distances - Nearest neighbor distances.
    :rtype: array-like
    """
    if x.shape[1] >= 20:
        tree = BallTree(x, metric='euclidean')
    else:
        tree = KDTree(x, metric='euclidean')
    nn = tree.query(x, k=2)[0][:, 1]
    return nn


def compute_d(x):
    R"""
    Computes the dimensionality of the data equal to the size of axis 1.

    :param x: Training instances.
    :type x: array-like
    """
    return self.x.shape[1]


def compute_mu(nn_distances, d):
    R"""
    Computes mu equal to the 1th percentile of :math:`mle(nn\text{_}distances, d) - 10`,
    where :math:`mle =
    \log(\text{gamma}(d/2 + 1)) - (d/2) \cdot \log(\pi) - d \cdot \log(nn\text{_}distances)`

    :param nn_distances: Observed nearest neighbor distances.
    :type nn_distances: array-like
    :type d: Dimensions.
    :type d: int
    :return: mu - The 1th percentile of :math:`mle(nn\text{_}distances, d) - 10`.
    :rtype: float
    """
    return quantile(mle(nn_distances, d), 0.01) - 10


def compute_ls(nn_distances):
    R"""
    Computes ls equal to the geometric mean of the nearest neighbor distances times a constant.

    :param nn_distances: Observed nearest neighbor distances.
    :type nn_distances: array-like
    :return: ls - The geometric mean of the nearest neighbor distances times a constant.
    :rtype: float
    """
    return exp(log(nn_distances).mean() + 3.1012095522922505)


def compute_cov_func(cov_func_curry, ls):
    R"""
    Computes the Gaussian process covariance function from its generator and length scale.

    :param cov_func_curry: The covariance function generator.
    :type cov_func_curry: function or type
    :param ls: Length scale of the covariance function.
    :type ls: float
    :return: cov_func - Gaussian process covariance function k(x, y) :math:`\rightarrow` float.
    :rtype: function
    """
    return cov_func_curry(ls)


def compute_initial_value(nn_distances, d, mu, L):
    R"""
    Computes the initial value for Maximum A Posteriori optimization with Ridge regression,
    such that the initial value :math:`z` minimizes
    :math:`||Lz + mu - mle(nn\text{_}distances, d)|| + ||z||`.

    :param nn_distances: Observed nearest neighbor distances.
    :type nn_distances: array-like
    :param d: Dimensions.
    :type d: int
    :param mu: Gaussian Process mean.
    :type mu: int
    :param L: A matrix such that :math:`L L^T \approx K`, where :math:`K` is the covariance matrix.
    :type L: array-like
    :return: initial_value - The argmin :math:`z`.
    :rtype: array-like
    """
    target = mle(nn_distances, d) - mu
    return Ridge(fit_intercept=False).fit(L, target).coef_