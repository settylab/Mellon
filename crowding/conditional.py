from jax.config import config
config.update("jax_enable_x64", True)
from jax.numpy import dot, sqrt, ones_like
from jax.numpy import sum as arraysum
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import solve_triangular


DEFAULT_SIGMA2 = 1e-6


def _full_conditional_mean(x, z, mu, L):
    """
    Builds the mean function of the conditioned full rank gp.

    :param x: Points.
    :type x: array-like
    :param z: pre-transformation values.
    :type z: array-like
    :param mu: Original Gaussian process mean.
    :type mu: float
    :param L: :math:`L` such that :math:`L L^T \approx K`, where K is the covariance matrix.
    :type L: array-like
    """
    weights = solve_triangular(L.T, z, lower=True)
    def mean(Xnew):
        Kus = cov_func(Xnew, x)
        return mu + dot(Kus, weights)
    return mean


def _standard_conditional_mean(xu, z, mu, cov_func):
    """
    Builds the mean function of the conditioned low rank gp, where rank
    is equal to the number of landmark points.
    
    :param x: Landmark points.
    :type x: array-like
    :param z: pre-transformation values.
    :type z: array-like
    :param mu: Original Gaussian process mean.
    :type mu: float
    :param cov_func: Gaussian process covariance function.
    :type cov_func: function
    """
    K = cov_func(xu, xu)
    L = cholesky(stabilize(K))
    weights = solve_triangular(L.T, z, lower=True)
    def mean(Xnew):
        Kus = cov_func(Xnew, xu)
        return mu + dot(Kus, weights)
    return mean


def _modified_conditional_mean(x, xu, log_densities_x, mu, cov_func, sigma2=DEFAULT_SIGMA2):
    """
    Builds the mean function of the conditioned low rank gp, where rank
    is less than the number of inducing points.
    
    :param x: Points.
    :type x: array-like
    :param xu: Landmark points.
    :type xu: array-like
    :param log_densities_x: Log density at each point in x.
    :type log_densities_x: array-like
    :param mu: Original Gaussian process mean.
    :type mu: float
    :param cov_func: Gaussian process covariance function.
    :type cov_func: function
    :param sigma2: White noise variance. Defaults to 1e-6.
    :type sigma2: float
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

def build_conditional_mean(rank, mu, x=None, xu=None, pre_transformation=None,
                           log_density_x=None, L=None, cov_func=None, sigma2=None):
    R"""
    Builds the mean function of the conditioned GP. Runs a different routine depending
    on the rank. Each routine requires the rank and mu, but each routine requires different
    optional arguments. If rank is equal to the number of data points, the mean is computed
    by conditioning on each data point. In this case, only x and pre_transformation
    must be provided. If rank is equal to the number of landmark points, the mean is
    computed by conditioning on each inducing point. In this case, only xu, pre_transformation,
    and cov_func must be provided. Otherwise, the mean is computed by inferring the mean at
    at the landmark points and conditioning on the inferred values. In this case, only x, xu,
    log_density_x, cov_func, and sigma2 must be provided.

    :param rank: The rank of the covariance matrix, or the percentage of the eigenvalues
        included in the eigenvectors used to construct L.
    :type rank: int
    :param mu: Original Gaussian process mean.
    :type mu: float
    :param x: Points.
    :type x: array-like
    :param xu: Landmark points.
    :type xu: array-like
    :param pre_transformation: pre-transformation values.
    :type pre_transformation: array-like
    :param log_densities_x: Log density at each point in x.
    :type log_densities_x: array-like
    :param L: L such that :math:`L L^T \approx K`, where K is the covariance matrix.
    :type L: array-like
    :param cov_func: Gaussian process covariance function.
    :type cov_func: function
    :param sigma2: White noise variance. Defaults to 1e-6.
    :type sigma2: float
    :return conditional_mean: Conditioned Gaussian process mean function.
    :rtype: function
    """
    if rank == x.shape[0]:
        conditional_mean = full_conditional_mean(x, pre_transformation, mu, L)
    elif rank == xu.shape[0]:
        conditional_mean = standard_conditional_mean(xu, pre_transformation, mu, cov_func)
    else:
        conditional_mean = modified_conditional_mean(x, xu, log_density_x,
                                                     mu, cov_func, sigma2=sigma2)
    return conditional_mean
    