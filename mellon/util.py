import sys
import logging

from jax.numpy import eye, log, pi, repeat, newaxis, tensordot, sqrt, maximum
from jax.numpy import sum as arraysum
from jax.scipy.special import gammaln
from jax import jit, vmap


DEFAULT_JITTER = 1e-6


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


def vector_map(fun, X, in_axis=0):
    """
    Applies jax just in time compilation and vmap to quickly evaluate a
    function for multiple input arrays.

    :param fun: The function to evaluate.
    :type fun: function
    :param X: Array of intput arrays.
    :type X: array-like
    :param in_axis: An integer, None, or (nested) standard Python container
        (tuple/list/dict) thereof specifying which input array axes to map over.
        S. documantation of jax.vmap.
    :return: Stacked results of the function calls.
    :rtype: array-like
    """
    vfun = vmap(jit(fun), in_axis)
    return vfun(X)


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
        cls.logger.setLevel(logging.CRITICAL + 1)

    @classmethod
    def on(cls):
        """Turn on logging."""
        cls.__new__(cls)
        cls.logger.setLevel(logging.INFO)
