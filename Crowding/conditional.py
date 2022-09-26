from jax.numpy import dot, sqrt, ones_like, eye
from jax.numpy import sum as arraysum
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import solve_triangular
from .util import stabilize, DEFAULT_JITTER


DEFAULT_SIGMA2 = 1e-6


def _full_conditional_mean(x, log_density_x, mu, cov_func,
                           jitter=DEFAULT_JITTER, sigma2=DEFAULT_SIGMA2):
    """
    Builds the mean function of the conditioned Gaussian process.

    :param x: The training instances.
    :type x: array-like
    :param log_densities_x: The log density at each point in x.
    :type log_densities_x: array-like
    :param mu: The original Gaussian process mean.
    :type mu: float
    :param cov_func: The Gaussian process covariance function.
    :type cov_func: function
    :param jitter: A small amount to add to the diagonal for stability. Defaults to 1e-6.
    :type jitter: float
    :param sigma2: The white noise variance. Defaults to 1e-6.
    :type sigma2: float
    :return: conditional_mean - The conditioned Gaussian process mean function.
    :rtype: function
    """
    K = cov_func(x, x)
    L = cholesky(stabilize(K, jitter=jitter+sigma2))
    weights = solve_triangular(L.T, solve_triangular(L, log_density_x, lower=True))
    def mean(Xnew):
        Kus = cov_func(Xnew, x)
        return mu + dot(Kus, weights)
    return mean


def _landmarks_conditional_mean(x, xu, log_density_x, mu, cov_func,
                                jitter=DEFAULT_JITTER, sigma2=DEFAULT_SIGMA2):
    """
    Builds the mean function of the conditioned low rank gp, where rank
    is less than the number of landmark points.

    :param x: The training instances.
    :type x: array-like
    :param xu: The landmark points.
    :type xu: array-like
    :param log_densities_x: The log density at each point in x.
    :type log_densities_x: array-like
    :param mu: The original Gaussian process mean.
    :type mu: float
    :param cov_func: The Gaussian process covariance function.
    :type cov_func: function
    :param jitter: A small amount to add to the diagonal for stability. Defaults to 1e-6.
    :type jitter: float
    :param sigma2: The white noise variance. Defaults to 1e-6.
    :type sigma2: float
    :return: conditional_mean - The conditioned Gaussian process mean function.
    :rtype: function
    """
    Kuu = cov_func(xu, xu)
    Kuf = cov_func(xu, x)
    Luu = cholesky(stabilize(Kuu))
    A = solve_triangular(Luu, Kuf, lower=True)
    Qffd = arraysum(A * A, 0)
    Lamd = ones_like(Qffd) * sigma2  # DTC
    A_l = A / Lamd
    L_B = cholesky(eye(xu.shape[0]) + dot(A_l, A.T))
    r = log_density_x - mu
    r_l = r / Lamd
    c = solve_triangular(L_B, dot(A, r_l), lower=True)
    z = solve_triangular(L_B.T, c)
    weights = solve_triangular(Luu.T, z)
    def mean(Xnew):
        Kus = cov_func(Xnew, xu)
        return mu + dot(Kus, weights)
    return mean