from jax.numpy import dot, sqrt, ones_like, eye
from jax.numpy import sum as arraysum
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import solve_triangular
from .util import stabilize


DEFAULT_SIGMA2 = 1e-6


def _full_conditional_mean(x, z, mu, L, cov_func):
    """
    Builds the mean function of the conditioned full rank Gaussian process.

    :param x: The training instances.
    :type x: array-like
    :param z: pre-transformation values.
    :type z: array-like
    :param mu: The original Gaussian process mean.
    :type mu: float
    :param L: A matrix such that :math:`L L^\top \approx K`, where :math:`K`
        is the covariance matrix.
    :type L: array-like
    :return: conditional_mean - The conditioned Gaussian process mean function.
    :rtype: function
    """
    weights = solve_triangular(L.T, z, lower=True)
    def mean(Xnew):
        Kus = cov_func(Xnew, x)
        return mu + dot(Kus, weights)
    return mean


def _standard_conditional_mean(x_, z, mu, cov_func):
    """
    Builds the mean function of the conditioned low rank gp.

    :param x_: The points to condition on.
    :type x_: array-like
    :param z: The optimized pre-transformation values.
    :type z: array-like
    :param mu: The original Gaussian process mean.
    :type mu: float
    :param cov_func: The Gaussian process covariance function.
    :type cov_func: function
    :return: conditional_mean - The conditioned Gaussian process mean function.
    :rtype: function
    """
    K = cov_func(x_, x_)
    L = cholesky(stabilize(K))
    weights = solve_triangular(L.T, z, lower=True)
    def mean(Xnew):
        Kus = cov_func(Xnew, x_)
        return mu + dot(Kus, weights)
    return mean


def _modified_conditional_mean(x, xu, log_densities_x, mu, cov_func, sigma2=DEFAULT_SIGMA2):
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
    r = log_densities_x - mu
    r_l = r / Lamd
    c = solve_triangular(L_B, dot(A, r_l), lower=True)
    z = solve_triangular(L_B.T, c)
    weights = solve_triangular(Luu.T, z)
    def mean(Xnew):
        Kus = cov_func(Xnew, xu)
        return mu + dot(Kus, weights)
    return mean

def compute_conditional_mean(rank, mu, cov_func, x=None, landmarks=None, pre_transformation=None,
                           log_density_x=None, L=None, sigma2=DEFAULT_SIGMA2):
    R"""
    Builds the mean function of the Gaussian process, conditioned on the log_density values.
    Returns a function equivalent to the predict function of the model. 

    :param rank: The rank of the covariance matrix, or the percentage of the eigenvalues
        included in the eigenvectors used to construct L.
    :type rank: int
    :param mu: The original Gaussian process mean.
    :type mu: float
    :param x: The training instances.
    :type x: array-like
    :param landmarks: The landmark points.
    :type landmarks: array-like
    :param pre_transformation: The optimized pre-transformation values.
    :type pre_transformation: array-like
    :param log_densities_x: The log density at each point in x.
    :type log_densities_x: array-like
    :param L: A matrix such that :math:`L L^\top \approx K`,
        where K is the covariance matrix.
    :type L: array-like
    :param cov_func: The Gaussian process covariance function.
    :type cov_func: function
    :param sigma2: The white noise variance. Defaults to 1e-6.
    :type sigma2: float
    :return: conditional_mean - The conditioned Gaussian process mean function.
    :rtype: function
    """
    if rank == x.shape[0]:
        conditional_mean = _full_conditional_mean(x, pre_transformation, mu, L, cov_func)
    elif rank > landmarks.shape[0]:
        conditional_mean = _standard_conditional_mean(x, pre_transformation, mu, cov_func)
    elif rank == landmarks.shape[0]:
        conditional_mean = _standard_conditional_mean(landmarks, pre_transformation, mu, cov_func)
    else:
        conditional_mean = _modified_conditional_mean(x, landmarks, log_density_x,
                                                      mu, cov_func, sigma2=sigma2)
    return conditional_mean
    