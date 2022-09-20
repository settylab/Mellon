from jax.config import config
config.update("jax_enable_x64", True)
from jax.numpy import log, pi, exp, quantile
from jax.numpy import sum as arraysum
from jax.scipy.special import gammaln
from jaxopt import ScipyMinimize


def _normal(k):
    R"""
    Builds the log pdf of :math:`z \sim \text{Normal}(0, I)`.

    :param k: Size of :math:`z`.
    :type k: int
    :return: The log pdf of :math:`z`.
    :rtype: function
    """
    def logpdf(z):
        return -(1/2)*arraysum(z**2) - (k/2)*log(2*pi)
    return logpdf


def _multivariate(mu, L):
    R"""
    Builds the transformation function from :math:`z \sim \text{Normal}(0, I)
    \rightarrow f \sim \text{Normal}(mu, K')`, where :math:`I` is the identity matrix
    and :math:`K \approx K' = L L^T`.

    :param mu: mean
    :type mu: float
    :param L: A matrix such that :math:`L L^T \approx K`.
    :type L: array-like
    :return: A function :math:`z \rightarrow f`.
    :rtype: function
    """
    def transform(z):
        return L.dot(z) + mu
    return transform


def _nearest_neighbors(r, d):
    """
    Returns the likelihood function of log densities :math:`p` given the observed
    distances :math:`L(p | r) = P(p | r)`, for number of dimensions :math:`d`.

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
    Builds the Bayesian loss function -(prior(:math:`z`) +
    likelihood(transform(:math:`z`))). Transform maps :math:`z \sim
    \text{Normal}(0, I) \rightarrow f \sim \text{Normal}(mu, K')`,
    where :math:`I` is the identity matrix and :math:`K \approx K' = L L^T`,
    where :math:`K` is the covariance matrix.
    Returns the loss function and transform.

    :param nn_distances: Observed nearest neighbor distances.
    :type nn_distances: function
    :param d: The dimensionality of the data
    :type d: int
    :param mu: Gaussian process mean.
    :type mu: float
    :param L: A matrix such that :math:`L L^T \approx K`, where :math:`K` is the
        covariance matrix.
    :type L: array-like
    :return: loss_func, transform_func - The Bayesian loss function and the
        transform function :math:`z \rightarrow f`.
    :rtype: function, function
    """
    k = L.shape[1]
    prior = _normal(k)
    likelihood = _nearest_neighbors(nn_distances, d)
    transform = _multivariate(mu, L)
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