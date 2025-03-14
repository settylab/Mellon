import logging
from collections import namedtuple
from functools import partial
from jax import random, vmap
from jax.numpy import log, pi, exp, stack, arange, sort, mean, zeros_like, any
from jax.numpy import sum as arraysum
from jax.scipy.special import gammaln
import jax.scipy.stats.norm as norm
import jax
from jax.example_libraries.optimizers import adam
from jaxopt import ScipyMinimize
from .conditional import (
    FullConditional,
    ExpFullConditional,
    FullConditionalTime,
    LandmarksConditional,
    ExpLandmarksConditional,
    LandmarksConditionalTime,
    LandmarksConditionalCholesky,
    ExpLandmarksConditionalCholesky,
    LandmarksConditionalCholeskyTime,
)
from .util import ensure_2d, DEFAULT_JITTER

logger = logging.getLogger("mellon")


DEFAULT_N_ITER = 100
DEFAULT_INIT_LEARN_RATE = 1e-1
DEFAULT_NUM_SAMPLES = 40
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
    Builds the transformation function from
    :math:`z \sim \text{Normal}(0, I) \rightarrow f \sim \text{Normal}(\mu, K')`,
    where :math:`I` is the identity matrix
    and :math:`K \approx K' = L L^\top`.

    :param mu: The Gaussian process mean :math:`\mu`.
    :type mu: float
    :param L: A matrix such that :math:`L L^\top \approx K`.
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


def _poisson(distances):
    """
    Returns the likelihood function of dimensionality and density given the
    observed k nearest-neighbor distances
    .
    :param distances: The observed nearest neighbor distances.
    :type distances: array-like
    :return: The likelihood function.
    :rtype: function
    """
    k = distances.shape[1]
    counts = arange(1, k + 1)

    ldist = sort(distances, axis=-1)
    ldist = log(ldist) + log(pi) / 2

    def V(d):
        """
        Return the log-volume of the n-sphere for the raidus related values in ldist.
        """
        return d * ldist - gammaln(d / 2 + 1)

    def logpdf(dims, log_dens):
        pred = log_dens[:, None] + V(dims[:, None])
        logp = pred * counts[None, :] - exp(pred) - gammaln(counts)[None, :]
        return arraysum(logp)

    return logpdf


def compute_transform(mu, L):
    R"""
    Computes a function transform that maps :math:`z \sim
    \text{Normal}(0, I) \rightarrow f \sim \text{Normal}(\mu, K')`,
    where :math:`I` is the identity matrix and :math:`K \approx K' = L L^\top`,
    where :math:`K` is the covariance matrix.

    :param mu: The Gaussian process mean:math:`\mu`.
    :type mu: float
    :param L: A matrix such that :math:`L L^\top \approx K`, where :math:`K` is the
        covariance matrix.
    :type L: array-like
    :return: transform - The transform function :math:`z \rightarrow f`.
    """
    return _multivariate(mu, L)


def compute_dimensionality_transform(mu_dim, mu_dens, L):
    R"""
    Computes a function transform that maps :math:`z \sim
    \text{Normal}(0, I) \rightarrow \log(f) \sim \text{Normal}(\mu, K')`,
    where :math:`I` is the identity matrix and :math:`K \approx K' = L L^\top`,
    where :math:`K` is the covariance matrix.

    :param mu: The Gaussian process mean :math:`\mu`.
    :type mu: float
    :param L: A matrix such that :math:`L L^\top \approx K`, where :math:`K` is the
        covariance matrix.
    :type L: array-like
    :return: transform - The transform function :math:`z \rightarrow f`.
    """

    dim_transform = _multivariate(mu_dim, L)
    dens_transform = _multivariate(mu_dens, L)

    def transform(z):
        dims, dens = z[0, :], z[1, :]
        return exp(dim_transform(dims)), dens_transform(dens)

    return transform


def compute_loss_func(nn_distances, d, transform, k):
    R"""
    Computes the Bayesian loss function -(prior(:math:`z`) +
    likelihood(transform(:math:`z`))).

    :param nn_distances: The observed nearest neighbor distances.
    :type nn_distances: array-like
    :param d: The dimensionality of the data.
    :type d: int
    :param transform:
        Maps :math:`z \sim \text{Normal}(0, I) \rightarrow f \sim \text{Normal}(\mu, K')`,
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


def compute_dimensionality_loss_func(distances, transform, k):
    R"""
    Computes the Bayesian loss function -(prior(:math:`z`) +
    likelihood(transform(:math:`z`))) for dimensionality inference.

    :param distances: The observed k nearest neighbor distances.
    :type distances: array-like
    :param transform:
        Maps :math:`z \sim \text{Normal}(0, I) \rightarrow \log(f) \sim \text{Normal}(\mu, K')`,
        where :math:`I` is the identity matrix and :math:`K \approx K' = L L^\top`,
        where :math:`K` is the covariance matrix.
    :type transform: function
    :param k: dimension of transform input
    :type k: int
    :return: loss_func - The Bayesian loss function
    :rtype: function, function
    """
    prior = _normal(k)
    likelihood = _poisson(distances)

    def loss_func(z):
        dims, log_dens = transform(z)
        return -(prior(z) + likelihood(dims, log_dens))

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
        :math:`z \sim \text{Normal}(0, I) \rightarrow f \sim \text{Normal}(\mu, K')`,
        where :math:`I` is the identity matrix and :math:`K \approx K' = L L^\top`,
        where :math:`K` is the covariance matrix.
    :type transform: function
    :return: log_density_x - The log density at the training points.
    """
    return transform(pre_transformation)


def compute_parameter_cov_factor(pre_transformation_std, L):
    R"""
    Computes :math:`\Sigma_L` the left factor of the covariance matrix
    of `log_density_x` the mean function of the Gaussian Process on the
    training data points. The uncertainty of the mean function comes from
    the uncertainty of the inferred model parameters quantified by
    `pre_transformation_std`.

    :param pre_transformation_std: Standard deviation of the parameters, e.g., as inferred by ADVI.
    :type pre_transformation_std: array-like
    :param L: A matrix such that :math:`L L^\top \approx K`, where :math:`K` is the
        covariance matrix of the Gaussian Process.
    :type L: array-like
    :return: sigma_L - The left factor of the covariance matrix of the transformed parameters.
    """
    return L * pre_transformation_std[None, :]


def compute_conditional(
    x,
    landmarks,
    pre_transformation,
    pre_transformation_std,
    y,
    mu,
    cov_func,
    L,
    Lp=None,
    sigma=0,
    jitter=DEFAULT_JITTER,
    y_is_mean=False,
    with_uncertainty=False,
):
    R"""
    Builds the mean function of the Gaussian process, conditioned on the
    function values (e.g., log-density) on x. Returns an instance of
    mellon.Predictor that acts as a function, defined on the whole domain of x.

    Parameters
    ----------
    x : array-like
        The training instances.
    landmarks : array-like or None
        The landmark points for fast sparse computation.
        Landmarks can be None if not using landmark points.
    pre_transformation : array-like or None
        The pre-transformed latent function representation.
    pre_transformation_std : array-like or None
        Standard deviation of the parameters, e.g., as inferred by ADVI.
    y : array-like
        The function values at each point in x.
    mu : float
        The original Gaussian process mean :math:`\mu`.
    cov_func : function
        The Gaussian process covariance function.
    L : array-like
        The matrix :math:`L` used to transform the latent function representation to
        thr Gaussin Process mean. Typically :math:`L L^\top \approx K`, where :math:`K` is the
        covariance matrix of the Gaussian Process.
    Lp : array-like, optional
        A matrix such that :math:`L_p L_p^\top = \Sigma_p`, where :math:`\Sigma_p` is the
        covariance matrix of the Gaussian Process on the inducing points.
    sigma : float, optional
        White noise variance, by default 0.
    jitter : float, optional
        A small amount to add to the diagonal for stability, by default 1e-6.
    y_is_mean : bool
        Wether to consider y the GP mean or a noise measurment
        subject to `sigma` or `y_cov_factor`. Has no effect if `L` is passed.
        Defaults to False.
    with_uncertainty : bool
        Wether to compute covariance functions and predictive uncertainty.
        Defaults to False.

    Returns
    -------
    mellon.Predictor
        The conditioned Gaussian process mean function.
    """

    if landmarks is None:
        logger.debug("Using FullConditional GP.")
        if with_uncertainty and pre_transformation_std is not None:
            y_cov_factor = compute_parameter_cov_factor(pre_transformation_std, L)
        else:
            y_cov_factor = None
        return FullConditional(
            x,
            y,
            mu,
            cov_func,
            Lp,
            sigma=sigma,
            jitter=jitter,
            y_cov_factor=y_cov_factor,
            y_is_mean=y_is_mean,
            with_uncertainty=with_uncertainty,
        )
    elif (
        pre_transformation is not None
        and pre_transformation.shape[0] == landmarks.shape[0]
    ):
        logger.debug("Using LandmarksConditionalCholesky GP.")
        landmarks = ensure_2d(landmarks)
        if pre_transformation_std is not None and sigma is not None and any(sigma > 0):
            raise ValueError(
                "One can specify either `sigma` or `pre_transformation_std` "
                "to describe uncertainty, but not both."
            )
        elif pre_transformation_std is not None:
            sigma = pre_transformation_std
        n_obs = x.shape[0]
        return LandmarksConditionalCholesky(
            landmarks,
            pre_transformation,
            mu,
            cov_func,
            n_obs,
            Lp,
            sigma=sigma,
            jitter=jitter,
            y_is_mean=y_is_mean,
            with_uncertainty=with_uncertainty,
        )
    else:
        logger.debug("Using LandmarksConditional GP.")
        landmarks = ensure_2d(landmarks)
        if with_uncertainty and pre_transformation_std is not None:
            y_cov_factor = compute_parameter_cov_factor(pre_transformation_std, L)
        else:
            y_cov_factor = None
        return LandmarksConditional(
            x,
            landmarks,
            y,
            mu,
            cov_func,
            L,
            sigma=sigma,
            jitter=jitter,
            y_cov_factor=y_cov_factor,
            y_is_mean=y_is_mean,
            with_uncertainty=with_uncertainty,
        )


def compute_conditional_times(
    x,
    landmarks,
    pre_transformation,
    pre_transformation_std,
    y,
    mu,
    cov_func,
    L,
    Lp,
    sigma=0,
    jitter=DEFAULT_JITTER,
    y_is_mean=False,
    with_uncertainty=False,
):
    R"""
    Builds the mean function of the Gaussian process, conditioned on the
    function values (e.g., log-density) on x, taking into account the associated
    times of each sample. Returns an instance of mellon.Predictor that acts as a
    function, defined on the whole domain of x, and can be evaluated at any given time.

    Parameters
    ----------
    x : array-like
        The training instances.
    landmarks : array-like or None
        The landmark points for fast sparse computation.
        Landmarks can be None if not using landmark points.
    pre_transformation : array-like or None
        The pre-transformed latent function representation.
    pre_transformation_std : array-like or None
        Standard deviation of the parameters, e.g., as inferred by ADVI.
    y : array-like
        The function values at each point in x.
    mu : float
        The original Gaussian process mean :math:`\mu`.
    cov_func : function
        The Gaussian process covariance function.
    L : array-like
        A matrix such that :math:`L L^\top \approx K`, where :math:`K` is the
        covariance matrix of the Gaussian Process.
    Lp : array-like, optional
        A matrix such that :math:`L_p L_p^\top = \Sigma_p`, where :math:`\Sigma_p` is the
        covariance matrix of the Gaussian Process on the inducing points.
    sigma : float, optional
        White noise variance, by default 0.
    jitter : float, optional
        A small amount to add to the diagonal for stability, by default 1e-6.
    y_is_mean : bool
        Wether to consider y the GP mean or a noise measurment
        subject to `sigma` or `y_cov_factor`. Has no effect if `L` is passed.
        Defaults to False.
    with_uncertainty : bool
        Wether to compute covariance functions and predictive uncertainty.
        Defaults to False.

    Returns
    -------
    mellon.Predictor
        The conditioned Gaussian process mean function that also accepts a 'times'
        argument to account for time-sensitive inferences.
    """

    if landmarks is None:
        logger.debug("Using FullConditional GP.")
        if pre_transformation_std is not None:
            y_cov_factor = compute_parameter_cov_factor(pre_transformation_std, L)
        else:
            y_cov_factor = None
        return FullConditionalTime(
            x,
            y,
            mu,
            cov_func,
            Lp,
            sigma=sigma,
            jitter=jitter,
            y_cov_factor=y_cov_factor,
            y_is_mean=y_is_mean,
            with_uncertainty=with_uncertainty,
        )
    elif (
        pre_transformation is not None
        and pre_transformation.shape[0] == landmarks.shape[0]
    ):
        logger.debug("Using LandmarksConditionalCholesky GP.")
        landmarks = ensure_2d(landmarks)
        if pre_transformation_std is not None and sigma is not None and any(sigma > 0):
            raise ValueError(
                "One can specify either `sigma` or `pre_transformation_std` "
                "to describe uncertainty, but not both."
            )
        elif pre_transformation_std is not None:
            sigma = pre_transformation_std
        n_obs = x.shape[0]
        return LandmarksConditionalCholeskyTime(
            landmarks,
            pre_transformation,
            mu,
            cov_func,
            n_obs,
            Lp,
            sigma=sigma,
            jitter=jitter,
            y_is_mean=y_is_mean,
            with_uncertainty=with_uncertainty,
        )
    else:
        logger.debug("Using LandmarksConditional GP.")
        landmarks = ensure_2d(landmarks)
        if pre_transformation_std is not None:
            y_cov_factor = compute_parameter_cov_factor(pre_transformation_std, L)
        else:
            y_cov_factor = None
        return LandmarksConditionalTime(
            x,
            landmarks,
            y,
            mu,
            cov_func,
            sigma=sigma,
            jitter=jitter,
            y_cov_factor=y_cov_factor,
            y_is_mean=y_is_mean,
            with_uncertainty=with_uncertainty,
        )


def compute_conditional_explog(
    x,
    landmarks,
    pre_transformation,
    pre_transformation_std,
    y,
    mu,
    cov_func,
    L,
    Lp,
    sigma=0,
    jitter=DEFAULT_JITTER,
    y_is_mean=False,
    with_uncertainty=False,
):
    R"""
    Builds the exp-mean function of the Gaussian process, conditioned on the
    function log-values (e.g., dimensionality) on x.
    Returns a function that is defined on the whole domain of x.

    Parameters
    ----------
    x : array-like
        The training instances.
    landmarks : array-like or None
        The landmark points for fast sparse computation.
        Landmarks can be None if not using landmark points.
    pre_transformation : array-like or None
        The pre-transformed latent function representation.
    pre_transformation_std : array-like or None
        Standard deviation of the parameters, e.g., as inferred by ADVI.
    y : array-like
        The function values at each point in x.
    mu : float
        The original Gaussian process mean :math:`\mu`.
    cov_func : function
        The Gaussian process covariance function.
    L : array-like
        The matrix :math:`L` used to transform the latent function representation to
        thr Gaussin Process mean. Typically :math:`L L^\top \approx K`, where :math:`K` is the
        covariance matrix of the Gaussian Process.
    Lp : array-like, optional
        A matrix such that :math:`L_p L_p^\top = \Sigma_p`, where :math:`\Sigma_p` is the
        covariance matrix of the Gaussian Process on the inducing points.
    sigma : float, optional
        White noise variance, by default 0.
    jitter : float, optional
        A small amount to add to the diagonal for stability, by default 1e-6.
    y_is_mean : bool
        Wether to consider y the GP mean or a noise measurment
        subject to `sigma` or `y_cov_factor`. Has no effect if `L` is passed.
        Defaults to False.
    with_uncertainty : bool
        Wether to compute covariance functions and predictive uncertainty.
        Defaults to False.

    Returns
    -------
    mellon.Predictor
        The conditioned Gaussian process mean function.
    """

    if landmarks is None:
        logger.debug("Using FullConditional GP.")
        if with_uncertainty and pre_transformation_std is not None:
            y_cov_factor = compute_parameter_cov_factor(pre_transformation_std, L)
        else:
            y_cov_factor = None
        y = log(y)
        return ExpFullConditional(
            x,
            y,
            mu,
            cov_func,
            Lp,
            sigma=sigma,
            jitter=jitter,
            y_cov_factor=y_cov_factor,
            y_is_mean=y_is_mean,
            with_uncertainty=with_uncertainty,
        )
    elif (
        pre_transformation is not None
        and pre_transformation.shape[0] == landmarks.shape[0]
    ):
        logger.debug("Using LandmarksConditionalCholesky GP.")
        landmarks = ensure_2d(landmarks)
        if pre_transformation_std is not None and sigma is not None and any(sigma > 0):
            raise ValueError(
                "One can specify either `sigma` or `pre_transformation_std` "
                "to describe uncertainty, but not both."
            )
        elif pre_transformation_std is not None:
            sigma = pre_transformation_std
        n_obs = x.shape[0]
        return ExpLandmarksConditionalCholesky(
            landmarks,
            pre_transformation,
            mu,
            cov_func,
            n_obs,
            Lp,
            sigma=sigma,
            jitter=jitter,
            y_is_mean=y_is_mean,
            with_uncertainty=with_uncertainty,
        )
    else:
        logger.debug("Using LandmarksConditional GP.")
        landmarks = ensure_2d(landmarks)
        if with_uncertainty and pre_transformation_std is not None:
            y_cov_factor = compute_parameter_cov_factor(pre_transformation_std, L)
        else:
            y_cov_factor = None
        y = log(y)
        return ExpLandmarksConditional(
            x,
            landmarks,
            y,
            mu,
            cov_func,
            sigma=sigma,
            jitter=jitter,
            y_cov_factor=y_cov_factor,
            y_is_mean=y_is_mean,
            with_uncertainty=with_uncertainty,
        )


def generate_gaussian_sample(rng, mean, log_std):
    """
    Generates a single sample from a multivariate Gaussian with diagonal covariance.

    :param rng: random number generator
    :param mean: mean of the Gaussian distribution
    :param log_std: logarithm of standard deviation of the Gaussian distribution
    :return: sample from the Gaussian distribution
    """
    return mean + exp(log_std) * random.normal(rng, mean.shape)


def calculate_gaussian_logpdf(x, mean, log_std):
    """
    Calculates the log probability density function of a multivariate Gaussian with diagonal covariance.

    :param x: value to evaluate the Gaussian on
    :param mean: mean of the Gaussian distribution
    :param log_std: logarithm of standard deviation of the Gaussian distribution
    :return: log pdf of the Gaussian distribution
    """
    return arraysum(vmap(norm.logpdf)(x, mean, exp(log_std)))


def calculate_elbo(logprob, rng, mean, log_std):
    """
    Calculates the single-sample Monte Carlo estimate of the variational lower bound (ELBO).

    :param logprob: log probability of the sample
    :param rng: random number generator
    :param mean: mean of the Gaussian distribution
    :param log_std: logarithm of standard deviation of the Gaussian distribution
    :return: ELBO estimate
    """
    sample = generate_gaussian_sample(rng, mean, log_std)
    return logprob(sample) - calculate_gaussian_logpdf(sample, mean, log_std)


def calculate_batch_elbo(logprob, rng, params, num_samples):
    """
    Calculates the average ELBO over a batch of random samples.

    :param logprob: log probability of the sample
    :param rng: random number generator key
    :param params: parameters of the Gaussian distribution
    :param num_samples: number of samples in the batch
    :return: batch ELBO
    """
    rngs = random.split(rng, num_samples)
    vectorized_elbo = vmap(partial(calculate_elbo, logprob), in_axes=(0, None, None))
    return mean(vectorized_elbo(rngs, *params))


def run_advi(
    loss_func,
    initial_parameters,
    n_iter=DEFAULT_N_ITER,
    init_learn_rate=DEFAULT_INIT_LEARN_RATE,
    nsamples=DEFAULT_NUM_SAMPLES,
    jit=DEFAULT_JIT,
):
    """
    Performs automatic differentiation variational inference (ADVI) to fit a
    Gaussian approximation to an intractable, unnormalized density.

    :param loss_func: function to calculate the loss
    :param initial_parameters: initial parameters for the optimization
    :param n_iter: number of iterations for the optimization (default: DEFAULT_N_ITER)
    :param init_learn_rate: The initial learn rate. Defaults to 1.
    :type init_learn_rate: float
    :return: parameters, standard deviations, and loss after the optimization
    :return: Results - A named tuple containing pre_transformation,
        pre_transformation_std, losses: The optimized parameters, the optimized
        standard deviations, and a history of ELBO values.
    :rtype: array-like, array-like, Object
    """

    def negative_logprob(x):
        return -loss_func(x)

    def objective(params, t):
        rng = random.PRNGKey(t)
        return -calculate_batch_elbo(negative_logprob, rng, params, nsamples)

    def learn_schedule(i):
        return exp(-1e-2 * i) * init_learn_rate

    init_mean, init_std = initial_parameters, -10 * zeros_like(initial_parameters)
    opt_init, opt_update, get_params = adam(learn_schedule)
    opt_state = opt_init((init_mean, init_std))

    def update(i, opt_state):
        params = get_params(opt_state)
        elbo, gradient = jax.value_and_grad(objective)(params, i)
        return opt_update(i, gradient, opt_state), elbo

    if jit:
        update = jax.jit(update)

    elbos = list()
    for t in range(n_iter):
        opt_state, elbo = update(t, opt_state)
        elbos.append(elbo.item())

    params, log_stds = get_params(opt_state)
    stds = exp(log_stds)

    Results = namedtuple("Results", "pre_transformation pre_transformation_std losses")
    return Results(params, stds, elbos)
