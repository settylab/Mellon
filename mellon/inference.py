from collections import namedtuple
from jax.numpy import log, pi, exp, stack
from jax.numpy import sum as arraysum
from jax.scipy.special import gammaln
import jax
from jax.example_libraries.optimizers import adam
from jaxopt import ScipyMinimize
from .conditional import (
    _full_conditional_mean,
    _full_conditional_mean_y,
    _landmarks_conditional_mean,
    _landmarks_conditional_mean_y,
)
from .util import DEFAULT_JITTER


DEFAULT_N_ITER = 100
DEFAULT_INIT_LEARN_RATE = 1
DEFAULT_OPTIMIZER = "L-BFGS-B"
DEFAULT_JIT = False


def _normal(k):
    R"""
    Builds the log pdf of :math:`z \sim \text{Normal}(0, I)`.

    :param k: The size of :math:`z`.
    :type k: int
    :return: The log pdf of :math:`z`.
    :rtype: function
    """

    def logpdf(z):
        return -(1 / 2) * arraysum(z**2) - (k / 2) * log(2 * pi)

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
    distances :math:`L(p | r) = P(r | p)`, for number of dimensions :math:`d`.
    :param r: The observed nearest neighbor distances.
    :type r: array-like
    :param d: The local dimensionality.
    :type d: int
    :return: The likelihood function.
    :rtype: function
    """
    const = (d * log(pi) / 2) - gammaln(d / 2 + 1)
    V = log(r) * d + const
    Vdr = log(d) + ((d - 1) * log(r)) + const

    def logpdf(log_density):
        A = exp(log_density + V)
        B = log_density + Vdr
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
    :param transform:
        Maps :math:`z \sim \text{Normal}(0, I) \rightarrow f \sim \text{Normal}(mu, K')`,
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

    return loss_func


def minimize_adam(
    loss_func,
    initial_value,
    n_iter=DEFAULT_N_ITER,
    init_learn_rate=DEFAULT_INIT_LEARN_RATE,
    jit=DEFAULT_JIT,
):
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
    :return: Results - A named tuple containing pre_transformation, opt_state, losses: The optimized
        parameters, final state of the optimizer, and history of loss values,
    :rtype: array-like, array-like, Object
    """

    def learn_schedule(i):
        return exp(-1e-2 * i) * init_learn_rate

    opt_init, opt_update, get_params = adam(learn_schedule)
    opt_state = opt_init(initial_value)
    val_grad = jax.value_and_grad(loss_func)
    if jit:
        val_grad = jax.jit(val_grad)

    def step(step, opt_state):
        value, grads = val_grad(get_params(opt_state))
        opt_state = opt_update(step, grads, opt_state)
        return value, opt_state

    losses = list()
    for i in range(n_iter):
        value, opt_state = step(i, opt_state)
        losses.append(value.item())
    pre_transformation = get_params(opt_state)
    losses = stack(losses)

    Results = namedtuple("Results", "pre_transformation opt_state losses")
    results = Results(pre_transformation, opt_state, losses)
    return results


def minimize_lbfgsb(loss_func, initial_value, jit=DEFAULT_JIT):
    R"""
    Minimizes function with a starting guess of initial_value.

    :param loss_func: Loss function to minimize.
    :type loss_func: function
    :param initial_value: Initial guess.
    :type initial_value: array-like
    :return: Results - A named tuple containing pre_transformation, opt_state,
        loss: The optimized parameters, final state of the optimizer, and the
        final loss value,
    :rtype: array-like, array-like, Object
    """
    opt = ScipyMinimize(fun=loss_func, method="L-BFGS-B", jit=jit).run(initial_value)
    Results = namedtuple("Results", "pre_transformation opt_state loss")
    results = Results(opt.params, opt.state, opt.state.fun_val.item())
    return results


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


def compute_conditional_mean(
    x,
    landmarks,
    y,
    mu,
    cov_func,
    sigma=0,
    jitter=DEFAULT_JITTER,
):
    R"""
    Builds the mean function of the Gaussian process, conditioned on the
    function values (e.g., log-density) on x.
    Returns a function that is defined on the whole domain of x.

    :param x: The training instances.
    :type x: array-like
    :param landmarks: The landmark points for fast sparse computation.
        Landmarks can be None if not using landmark points.
    :type landmarks: array-like
    :param y: The function values at each point in x.
    :type y: array-like
    :param mu: The original Gaussian process mean.
    :type mu: float
    :param cov_func: The Gaussian process covariance function.
    :type cov_func: function
    :param sigma: White moise veriance. Defaults to 0.
    :type sigma: float
    :param jitter: A small amount to add to the diagonal for stability. Defaults to 1e-6.
    :type jitter: float
    :return: conditional_mean - The conditioned Gaussian process mean function.
    :rtype: function
    """
    if landmarks is None:
        return _full_conditional_mean(
            x,
            y,
            mu,
            cov_func,
            jitter=jitter,
        )
    else:
        if len(landmarks.shape) < 2:
            landmarks = landmarks[:, None]
        return _landmarks_conditional_mean(
            x,
            landmarks,
            y,
            mu,
            cov_func,
            sigma=sigma,
            jitter=jitter,
        )


def compute_conditional_mean_y(
    x,
    landmarks,
    Xnew,
    mu,
    cov_func,
    sigma=0,
    jitter=DEFAULT_JITTER,
):
    R"""
    Builds the mean function of the Gaussian process, conditioned on the
    function values (e.g., log-density) on x, and for fixed
    output locations Xnew and therefor flexible output values y.

    :param x: The training instances.
    :type x: array-like
    :param landmarks: The landmark points for fast sparse computation.
        Landmarks can be None if not using landmark points.
    :type landmarks: array-like
    :param Xnew: The output locations.
    :type Xnew: array-like
    :param mu: The original Gaussian process mean.
    :type mu: float
    :param cov_func: The Gaussian process covariance function.
    :type cov_func: function
    :param sigma: White moise veriance. Defaults to 0.
    :type sigma: float
    :param jitter: A small amount to add to the diagonal for stability. Defaults to 1e-6.
    :type jitter: float
    :return: conditional_mean - The conditioned Gaussian process mean function.
    :rtype: function
    """
    if landmarks is None:
        return _full_conditional_mean_y(
            x,
            Xnew,
            mu,
            cov_func,
            jitter=jitter,
        )
    else:
        if len(landmarks.shape) < 2:
            landmarks = landmarks[:, None]
        return _landmarks_conditional_mean_y(
            x,
            landmarks,
            Xnew,
            mu,
            cov_func,
            sigma=sigma,
            jitter=jitter,
        )
