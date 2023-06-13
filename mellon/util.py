import sys
import logging

from jax.config import config as jaxconfig
from jax.numpy import (
    eye,
    log,
    pi,
    repeat,
    newaxis,
    tensordot,
    sqrt,
    maximum,
    triu_indices,
    sort,
    ones,
    arange,
    concatenate,
)
from jax.numpy import sum as arraysum
from jax.numpy.linalg import norm, lstsq, matrix_rank
from jax.scipy.special import gammaln
from jax import vmap
from sklearn.neighbors import BallTree, KDTree


DEFAULT_JITTER = 1e-6
DEFAULT_RANK_TOL = 5e-1


def stabilize(A, jitter=DEFAULT_JITTER):
    R"""
    Add a small jitter to the diagonal for numerical stability.

    :param A: A square matrix.
    :param jitter: The amount to add to the diagonal. Defaults to 1e-6.
    :type jitter: float
    :return: :math:`A'` - The matrix :math:`A` with a small jitter added to the diagonal.
    :rtype: array-like
    """
    n = A.shape[0]
    return A + eye(n) * jitter


def mle(nn_distances, d):
    R"""
    Nearest Neighbor distribution maximum likelihood estimate for log density
    given observed nearest neighbor distances :math:`nn\text{_}distances` in
    dimensions :math:`d`: :math:`mle = \log(\text{gamma}(d/2 + 1)) - (d/2) \cdot \log(\pi) -
    d \cdot \log(nn\text{_}distances)`

    :param nn_distances: The observed nearest neighbor distances.
    :type nn_distances: array-like
    :param d: The local dimensionality of the data.
    :type d: int
    :return: :math:`mle` - The maximum likelihood estimate at each point.
    :rtype: array-like
    """
    return gammaln(d / 2 + 1) - (d / 2) * log(pi) - d * log(nn_distances)


def distance(x, y):
    """
    Computes the distances between each point in x and y.

    :param x: A set of points.
    :type x: array-like
    :param y: A set of points.
    :type y: array-like
    :return: distances - The distance between each point in x and y.
    :rtype: array-like
    """
    n = x.shape[0]
    m = y.shape[0]
    xx = repeat(arraysum(x * x, axis=1)[:, newaxis], m, axis=1)
    yy = repeat(arraysum(y * y, axis=1)[newaxis, :], n, axis=0)
    xy = tensordot(x, y, (1, 1))
    sq = xx - 2 * xy + yy + 1e-12
    return sqrt(maximum(sq, 0))


def test_rank(input, tol=DEFAULT_RANK_TOL, threshold=None):
    """
    Inspects the approximate rank of the transformation matrix L. The input can be the matrix itself or an object
    containing L as an attribute. A high rank indicates a potentially insufficient latent representation,
    suggesting a need for a more complex transformation matrix. Also allows logging based on a rank fraction threshold.

    :param input: The matrix L or an object containing it as an attribute.
    :type input: array-like or mellon estimator object
    :param tol: The rank calculation tolerance, defaults to {DEFAULT_RANK_TOL}.
    :type tol: float, optional
    :param threshold: If provided, logs a message based on the rank fraction.
    :type threshold: float, optional
    :return: The approximate rank of the matrix.
    :rtype: int
    :raises ValueError: If the input matrix is not 2D.
    """
    if hasattr(input, "shape"):
        L = input
    elif hasattr(input, "L"):
        L = input.L
        if L is None:
            raise AttributeError(
                "Matrix L is not found in the estimator object. Consider running `.prepare_inference()`."
            )
    else:
        raise TypeError(
            "Input must be either a matrix or a mellon enstimator with a transformation L."
        )

    if len(L.shape) != 2:
        raise ValueError("Matrix L must be 2D.")

    approx_rank = matrix_rank(L, tol=tol)
    max_rank = min(L.shape)
    rank_fraction = approx_rank / max_rank

    if threshold is not None:
        logger = Log()
        if rank_fraction > threshold:
            logger.warning(
                f"High approx. rank fraction ({rank_fraction:.1%}). "
                "Consider increasing 'n_landmarks'."
            )
        else:
            logger.info(
                f"Rank fraction ({rank_fraction:.1%}, lower is better) is "
                "within acceptable range. "
                "Current settings should provide satisfactory model performance."
            )
    else:
        print(
            f"The approx. rank fraction is {rank_fraction:.1%} "
            f"({approx_rank:,} of {max_rank:,}). Lower is better."
        )

    return approx_rank.item()


def local_dimensionality(x, k=30, x_query=None, neighbor_idx=None):
    """
    Compute an estimate of the local fractal dimension of a data set using nearest neighbors.

    :param x: The input samples.
    :type x: array-like of shape (n_samples, n_features)
    :param k: The number of neighbors to consider, defaults to 30.
    :type k: int, optional
    :param x_query: The points at which to compute the local fractal dimension.
        If None, use x itself, defaults to None.
    :type x_query: array-like of shape (n_queries, n_features), optional
    :param neighbor_idx: The indices of the neighbors for each query point.
        If None, these are computed using a nearest neighbor search, defaults to None.
    :type neighbor_idx: array-like of shape (n_queries, k), optional
    :return: The estimated local fractal dimension at each query point.
    :rtype: array-like of shape (n_queries,)

    This function computes the local fractal dimension of a dataset at query points.
    It uses nearest neighbors and fits a line in log-log space to estimate the fractal dimension.
    """
    if neighbor_idx is None:
        if x_query is None:
            x_query = x
        if x.shape[1] >= 20:
            tree = BallTree(x, metric="euclidean")
        else:
            tree = KDTree(x, metric="euclidean")
        neighbors = x[tree.query(x_query, k=k)[1]]
    else:
        neighbors = x[neighbor_idx]
    i, j = triu_indices(k, k=1)
    neighbor_distances = norm(neighbors[..., i, :] - neighbors[..., j, :], axis=-1)
    neighborhood_distances = sort(neighbor_distances, axis=-1)

    kc2 = k * (k - 1) // 2
    A = concatenate(
        [log(neighborhood_distances)[..., None], ones((x_query.shape[0], kc2, 1))],
        axis=-1,
    )
    y = log(arange(1, kc2 + 1))[:, None]

    vreg = vmap(lstsq, in_axes=(0, None))
    w, _, _, _ = vreg(A, y)
    return w[:, 0, 0]


class Log(object):
    """Access the Mellon logging/verbosity. Log() returns the singelon logger and
    Log.off() and Log.on() disable or enable logging respectively."""

    def __new__(cls):
        """Return the singelton Logger."""
        if not hasattr(cls, "logger"):
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            cls.handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("[%(asctime)s] [%(levelname)-8s] %(message)s")
            cls.handler.setFormatter(formatter)
            logger.addHandler(cls.handler)
            logger.propagate = False
            cls.logger = logger
        return cls.logger

    @classmethod
    def off(cls):
        """Turn off all logging."""
        cls.__new__(cls)
        cls.logger.setLevel(logging.WARNING + 1)

    @classmethod
    def on(cls):
        """Turn on logging."""
        cls.__new__(cls)
        cls.logger.setLevel(logging.INFO)


def set_jax_config(enable_x64=True, platform_name="cpu"):
    """
    Sets up the JAX configuration with the specified settings.

    Parameters
    ----------
    enable_x64 : bool, optional
        Whether to enable 64-bit (double precision) computations in JAX.
        Defaults to True.
    platform_name : str, optional
        The platform name to use in JAX ('cpu', 'gpu', or 'tpu').
        Defaults to 'cpu'.
    """
    jaxconfig.update("jax_enable_x64", enable_x64)
    jaxconfig.update("jax_platform_name", platform_name)
