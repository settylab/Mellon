from jax.config import config
config.update("jax_enable_x64", True)
from jax.numpy import log, pi, exp, quantile
from jax.numpy import sum as arraysum
from jax.scipy.special import gammaln
from jaxopt import ScipyMinimize


def _normal(k):
    R"""
    Builds the log pdf of z ~ Normal(0, I).
    
    :param k: Size of z.
    :type k: int
    :return: The log pdf of z.
    :rtype: function
    """
    def logpdf(z):
        return -(1/2)*arraysum(z**2) - (k/2)*log(2*pi)
    return logpdf


def _multivariate(mu, L):
    R"""
    Builds the transformation function from z ~ Normal(0, I) to
    f ~ Multivariate_Normal(mu, K'), where I is the identity matrix
    and :math:`K \approx K' = L L^T`.

    :param mu: mean
    :type mu: float
    :param L: A matrix such that :math:`K \approx L L^T`.
    :type L: array-like
    :return: A function z -> f.
    :rtype: function
    """
    def transform(z):
        return L.dot(z) + mu
    return transform


def _nearest_neighbors(r, d):
    """
    Returns the likelihood function of log densities p given the observed
    distances L(p | r) = P(p | r), for number of dimensions d.
    
    :param r: Observed nearest neighbor distances.
    :type r: array-like
    :param d: Number of dimensions.
    :type d: int
    :return: The likelihood function.
    :rtype: function
    """
    constant1 = pi**(d/2) / exp(gammaln(d/2 + 1))
    constant2 = log(d) + (d * log(pi) / 2) - gammaln(d/2 + 1)
    def volume(r):
        return constant1 * (r**d)
    def log_dvolume_dr(r):
        return constant2 + ((d-1) * log(r))
    def logpdf(log_density):
        # log-probability-density function for distance r
        A = exp(log_density) * volume(r)
        B = log_density + log_dvolume_dr(r)
        return arraysum(B - A)
    return logpdf


def inference_functions(nn_distances, d, mu, L):
    R"""
    Builds the Bayesian loss function -(prior(z) + likelihood(transform(z))).
    Transform maps z ~ Normal(0, I) -> f ~ Multivariate_Normal(mu, K'), where I
    is the identity matrix and :math:`K \approx K' = L L^T`.
    
    Returns the loss function and transform.

    :param nn_distances: The observed nearest neighbor distances.
    :type nn_distances: function
    :param d: The dimensionality of the data
    :type d: int
    :param mu: The Gaussian process mean.
    :type mu: float
    :param L: A matrix such that :math:`K \approx L L^T`, where K is the
        covariance matrix.
    :type L: array-like
    :return loss_func, transform_func: The Bayesian loss function and the
        transform function z -> f.
    :rtype: function, function
    """
    k = L.shape[1]
    prior = normal(k)
    likelihood = nearest_neighbors(nn_distances, d)
    transform = multivariate(mu, L)
    def loss_func(z):
        return -(prior(z) + likelihood(transform(z)))
    return loss_func, transform

def run_inference(loss_func, initial_value):
    R"""
    Minimizes function with a starting guess of initial_value.
    
    :param loss_func: Loss function to minimize.
    :type loss_func: function
    :param initial_value: Initial guess.
    :type initial_value: array-like
    :return: Results of the optimization.
    :rtype: OptStep
    """
    return ScipyMinimize(fun=loss_func, method="L-BFGS-B").run(initial_value)