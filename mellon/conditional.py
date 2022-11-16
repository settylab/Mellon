from jax.numpy import dot, ones_like, eye, square
from jax.numpy import sum as arraysum
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import solve_triangular
from .util import stabilize, DEFAULT_JITTER


def _full_conditional_mean(
    x, y, mu, cov_func, sigma=0, jitter=DEFAULT_JITTER,
):
    """
    Builds the mean function of the conditioned Gaussian process.

    :param x: The training instances.
    :type x: array-like
    :param y: The function value at each point in x.
    :type y: array-like
    :param mu: The original Gaussian process mean.
    :type mu: float
    :param cov_func: The Gaussian process covariance function.
    :type cov_func: function
    :param sigma: White moise standard deviation. Defaults to 0.
    :type sigma: float
    :param jitter: A small amount to add to the diagonal for stability. Defaults to 1e-6.
    :type jitter: float
    :return: conditional_mean - The conditioned Gaussian process mean function.
    :rtype: function
    """
    sigma2 = square(sigma)
    K = cov_func(x, x)
    sigma2 = max(sigma2, jitter)
    L = cholesky(stabilize(K, jitter=sigma2))
    weights = solve_triangular(L.T, solve_triangular(L, y, lower=True))

    def mean(Xnew):
        Kus = cov_func(Xnew, x)
        return mu + dot(Kus, weights)

    return mean


def _landmarks_conditional_mean(
    x, xu, y, mu, cov_func, sigma=0, jitter=DEFAULT_JITTER,
):
    """
    Builds the mean function of the conditioned low rank gp, where rank
    is less than the number of landmark points.

    :param x: The training instances.
    :type x: array-like
    :param xu: The landmark points.
    :type xu: array-like
    :param y: The function value at each point in x.
    :type y: array-like
    :param mu: The original Gaussian process mean.
    :type mu: float
    :param cov_func: The Gaussian process covariance function.
    :type cov_func: function
    :param sigma: White moise standard deviation. Defaults to 0.
    :type sigma: float
    :param jitter: A small amount to add to the diagonal for stability. Defaults to 1e-6.
    :type jitter: float
    :return: conditional_mean - The conditioned Gaussian process mean function.
    :rtype: function
    """
    sigma2 = square(sigma)
    Kuu = cov_func(xu, xu)
    Kuf = cov_func(xu, x)
    Luu = cholesky(stabilize(Kuu, jitter))
    A = solve_triangular(Luu, Kuf, lower=True)
    sigma2 = max(sigma2, jitter)
    L_B = cholesky(stabilize(dot(A, A.T), sigma2))
    r = y - mu
    c = solve_triangular(L_B, dot(A, r), lower=True)
    z = solve_triangular(L_B.T, c)
    weights = solve_triangular(Luu.T, z)

    def mean(Xnew):
        Kus = cov_func(Xnew, xu)
        return mu + dot(Kus, weights)

    return mean
