from jax.numpy import log, pi, exp, quantile
from jax.numpy import sum as arraysum
from jax.scipy.special import gammaln
from jaxopt import ScipyMinimize


DEFAULT_N_ITER = 100
DEFAULT_INIT_LEARN_RATE = 1


def _normal(k):
    R"""
    Builds the log pdf of :math:`z \sim \text{Normal}(0, I)`.

    :param k: The size of :math:`z`.
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
    and :math:`K \approx K' = L L^\top`.

    :param mu: The Gaussian process mean.
    :type mu: float
    :param L: A matrix such that :math:`L L^\top
    \approx K`.
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

    :param r: The observed nearest neighbor distances.
    :type r: array-like
    :param d: The local dimensionality.
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

def compute_transform(mu, L):
    R"""
    Computes a function transform that maps :math:`z \sim
    \text{Normal}(0, I) \rightarrow f \sim \text{Normal}(mu, K')`,
    where :math:`I` is the identity matrix and :math:`K \approx K' = L L^\top`,
    where :math:`K` is the covariance matrix.

    :param mu: The Gaussian process mean.
    :type mu: float
    :param L: A matrix such that :math:`L L^\top \approx K`, where :math:`K` is the
        covariance matrix.
    :type L: array-like
    :return: transform - The transform function :math:`z \rightarrow f`.
    """
    return _multivariate(mu, L)


def compute_loss_func(nn_distances, d, transform, k):
    R"""
    Computes the Bayesian loss function -(prior(:math:`z`) +
    likelihood(transform(:math:`z`))). 

    :param nn_distances: The observed nearest neighbor distances.
    :type nn_distances: array-like
    :param d: The dimensionality of the data.
    :type d: int
    :param transform: Maps :math:`z \sim \text{Normal}(0, I) \rightarrow f \sim \text{Normal}(mu, K')`,
        where :math:`I` is the identity matrix and :math:`K \approx K' = L L^\top`,
        where :math:`K` is the covariance matrix.
    :type transform: function
    :param k: dimension of transform input
    :type k: int
    :return: loss_func - The Bayesian loss function
    :rtype: function, function
    """
    prior = _normal(k)
    likelihood = _nearest_neighbors(nn_distances, d)
    def loss_func(z):
        return -(prior(z) + likelihood(transform(z)))
    return loss_func, transform


def run_inference(loss_func, initial_value, n_iter=DEFAULT_N_ITER, \
                  init_learn_rate=DEFAULT_INIT_LEARN_RATE):
    R"""
    Minimizes function with a starting guess of initial_value using
    adam and exponentially decaying learning rate.

    :param loss_func: The loss function to minimize.
    :type loss_func: function
    :param initial_value: The initial guess.
    :type initial_value: array-like
    :param n_iter: The number of optimization iterations. Defaults to 100.
    :type n_iter: integer
    :param init_learn_rate: The initial learn rate. Defaults to 1.
    :type init_learn_rate: float
    :return: pre_transformation, loss, opt_state - The optimized parameters, history of
        loss values, and final state of the optimizer.
    :rtype: array-like, array-like, Object
    """
    def learn_schedule(i):
        return jnp.exp(-1e-2 * i) * init_learn_rate

    opt_init, opt_update, get_params = adam(learn_schedule)
    opt_state = opt_init(initial_value)
    val_grad = jax.value_and_grad(loss_func)

    def step(step, opt_state):
        value, grads = val_grad(get_params(opt_state))
        opt_state = opt_update(step, grads, opt_state)
        return value, opt_state

    losses = list()
    for i in range(n_iter):
        value, opt_state = step(i, opt_state)
        losses.append(value)
    pre_transformation = get_params(opt_state)
    losses = jnp.stack(losses)

    return pre_transformation, (opt_state, losses)


def compute_log_density_x(pre_transformation, transform):
    R"""
    Computes the log density at the training points.

    :param pre_transformation: :math:`z \sim \text{Normal}(0, I)`
    :type pre_transformation: array-like
    :param transform: A function
        :math:`z \sim \text{Normal}(0, I) \rightarrow f \sim \text{Normal}(mu, K')`,
        where :math:`I` is the identity matrix and :math:`K \approx K' = L L^\top`,
        where :math:`K` is the covariance matrix.
    :type transform: function
    :return: log_density_x - The log density at the training points.
    """
    return transform(pre_transformation)