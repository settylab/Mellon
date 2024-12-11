import logging
from jax import vmap
from jax.numpy import dot, square, isnan, any, eye, zeros, arange, ndim, isscalar
from jax.numpy import sum as arraysum
from jax.numpy import diag as diagonal
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import solve_triangular, solve
from .util import ensure_2d, stabilize, DEFAULT_JITTER, add_variance
from .base_predictor import Predictor, ExpPredictor, PredictorTime
from .decomposition import DEFAULT_SIGMA


logger = logging.getLogger("mellon")


def _get_L(x, cov_func, jitter=DEFAULT_JITTER, y_cov_factor=None):
    K = cov_func(x, x)
    K = add_variance(K, y_cov_factor, jitter=jitter)
    L = cholesky(K)
    if any(isnan(L)):
        message = (
            f"Covariance not positively definite with jitter={jitter}. "
            "Consider increasing the jitter for numerical stabilization."
        )
        logger.error(message)
        raise ValueError(message)
    return L


def _check_covariance(obj):
    if not hasattr(obj, "L"):
        raise ValueError(
            "The predictor was computed without covariance. "
            "Recompute setting `with_uncertainty=True.`"
        )


def _check_uncertainty(obj):
    if not hasattr(obj, "W"):
        raise ValueError(
            "The predictor was computed without uncertainty, e.g., using ADVI. "
            "Recompute setting `with_uncertainty=True.` and define `pre_transformation_std`"
            ", e.g., by using `optimizer='advi'`."
        )


def _sigma_to_y_cov_factor(sigma, y_cov_factor, n):
    if sigma is None and y_cov_factor is None:
        message = (
            "No input uncertainty specified. Make sure to set `sigma` or `pre_transformation_std`, "
            'e.g., by using `optimizer="advi", to quantify uncertainty of the prediction.'
        )
        logger.error(message)
        raise ValueError(message)
    if y_cov_factor is not None and sigma is not None and any(sigma > 0):
        raise ValueError(
            "One can specify either `sigma` or `y_cov_factor` to describe input noise, but not both."
        )

    if y_cov_factor is not None:
        return y_cov_factor

    sigma_ndim = ndim(sigma)
    if sigma_ndim == 0:
        y_cov_factor = eye(n) * sigma
    elif sigma_ndim == 1:
        y_cov_factor = diagonal(sigma)
    elif sigma_ndim > 1:
        # Extend sigma to higher dimensions, adding a leading dimension for the diagonal
        y_cov_factor = zeros((n,) + sigma.shape)

        def update_diag(i, ycf, val):
            return ycf.at[i, ...].set(val)

        y_cov_factor = vmap(update_diag, in_axes=(0, 0, 0), out_axes=0)(
            arange(n), y_cov_factor, sigma
        )
    else:
        raise ValueError(f"Unsupported `sigma` dimensions `{sigma_ndim}`.")

    return y_cov_factor


def _process_sigma(sigma, r, A, jitter=DEFAULT_JITTER):
    """
    Helper function to interpret and process sigma based on its shape and the shape of r.

    Args:
        sigma: Noise or covariance descriptor (scalar, element-wise, or matrix).
        r: Residual or target vector/matrix.
        A: Design or transformation matrix.

    Returns:
        Tuple (r_l, A_l):
            - r_l: Adjusted r vector/matrix.
            - A_l: Adjusted A matrix.

    Raises:
        NotImplementedError: If the sigma configuration is unsupported.
    """
    if isscalar(sigma) or sigma.shape == r.shape and r.ndim <= 1:
        logger.info("Sigma interpreted as element-wise standard deviation.")
        sigma2 = square(sigma)
        r_l = r / sigma2
        A_l = A / sigma2
    elif sigma.shape == r.shape and r.ndim > 1:
        logger.error("Sigma as distinct noise per output is not implemented.")
        raise NotImplementedError(
            "FunctionEstimator not implemented for multiple noises."
        )
    elif sigma.shape == (r.shape[0],) + r.shape and r.ndim > 1:
        logger.error(
            "Sigma as distinct covariance matrix per output is not implemented."
        )
        raise NotImplementedError(
            "FunctionEstimator not implemented for multiple covariance matrices."
        )
    elif sigma.shape == (r.shape[0], r.shape[0]):
        logger.info("Sigma interpreted as full covariance matrix.")
        L_s = cholesky(stabilize(sigma, jitter))
        r_l = solve_triangular(L_s.T, solve_triangular(L_s, r, lower=True))
        A_l = solve_triangular(L_s.T, solve_triangular(L_s, A, lower=True))
    else:
        raise ValueError("Unsupported sigma configuration.")

    return r_l, A_l


class _FullConditional:
    def __init__(
        self,
        x,
        y,
        mu,
        cov_func,
        L=None,
        sigma=DEFAULT_SIGMA,
        jitter=DEFAULT_JITTER,
        y_cov_factor=None,
        y_is_mean=False,
        with_uncertainty=False,
    ):
        R"""
        The mean function of the conditioned Gaussian process (GP).

        :param x: The training instances.
        :type x: array-like
        :param y: The function value at each point in x.
        :type y: array-like
        :param mu: The original GP mean.
        :type mu: float
        :param cov_func: The GP covariance function.
        :type cov_func: function
        :param L : A matrix such that :math:`L L^\top \approx K`, where :math:`K` is the
            covariance matrix of the GP.
        :type L : array-like or None
        :param sigma: Noise standard deviation of the data we condition on. Defaults to 0.
        :type sigma: float
        :param jitter: A small amount to add to the diagonal for stability. Defaults to 1e-6.
        :type jitter: float
        :param y_cov_factor: A matrix :math:`\Sigma_L` such that
            :math:`\Sigma_L\cdot\Sigma_L` is the covaraince of `y`.
            Only required if `with_uncertainty=True`. Defaults to None.
        :type y_cov_factor: array-like
        :param y_is_mean: Wether to consider y the GP mean or a noise measurment
            subject to `sigma` or `y_cov_factor`. Has no effect if `L` is passed.
            Defaults to False.
        :type y_is_mean: bool
        :param with_uncertainty: Wether to compute covariance functions and
            predictive uncertainty. Defaults to False.
        :type with_uncertainty: bool
        :return: conditional_mean - The conditioned GP mean function.
        :rtype: function
        """
        x = ensure_2d(x)

        if L is None:
            logger.info("Recomputing covariance decomposition for predictive function.")
            if y_is_mean:
                logger.debug("Assuming y is the mean of the GP.")
                L = _get_L(x, cov_func, jitter)
            else:
                logger.debug("Assuming y is not the mean of the GP.")
                y_cov_factor = _sigma_to_y_cov_factor(sigma, y_cov_factor, x.shape[0])
                sigma = None
                L = _get_L(x, cov_func, jitter, y_cov_factor)
        r = y - mu
        weights = solve_triangular(L.T, solve_triangular(L, r, lower=True))

        self.cov_func = cov_func
        self.x = x
        self.weights = weights
        self.mu = mu
        self.jitter = jitter
        self.n_input_features = x.shape[1]
        self.n_obs = x.shape[0]

        self._state_variables = {"x", "weights", "mu", "jitter"}

        if not with_uncertainty:
            return

        self.L = L
        self._state_variables.add("L")

        y_cov_factor = _sigma_to_y_cov_factor(sigma, y_cov_factor, x.shape[0])

        W = solve_triangular(L.T, solve_triangular(L, y_cov_factor, lower=True))
        self.W = W
        self._state_variables.add("W")

    def _mean(self, Xnew):
        cov_func = self.cov_func
        x = self.x
        weights = self.weights
        mu = self.mu

        Kus = cov_func(Xnew, x)
        return mu + dot(Kus, weights)

    def _covariance(self, Xnew, diag=True):
        _check_covariance(self)
        x = self.x
        cov_func = self.cov_func
        L = self.L

        Kus = cov_func(x, Xnew)
        A = solve_triangular(L, Kus, lower=True)
        if diag:
            Kss = cov_func.diag(Xnew)
            var = Kss - arraysum(square(A), axis=0)
            return var
        else:
            Kss = cov_func(Xnew, Xnew)
            cov = Kss - dot(A.T, A)
            return cov

    def _mean_covariance(self, Xnew, diag=True):
        _check_uncertainty(self)
        cov_func = self.cov_func
        x = self.x
        W = self.W

        Kus = cov_func(Xnew, x)
        cov_L = Kus @ W

        if diag:
            var = arraysum(cov_L * cov_L, axis=1)
            return var
        else:
            cov = cov_L @ cov_L.T
            return cov


class FullConditional(_FullConditional, Predictor):
    pass


class ExpFullConditional(_FullConditional, ExpPredictor):
    pass


class FullConditionalTime(_FullConditional, PredictorTime):
    pass


class _LandmarksConditional:
    def __init__(
        self,
        x,
        xu,
        y,
        mu,
        cov_func,
        L=None,
        Lp=None,
        sigma=DEFAULT_SIGMA,
        jitter=DEFAULT_JITTER,
        y_cov_factor=None,
        y_is_mean=False,
        with_uncertainty=False,
    ):
        R"""
        The mean function of the conditioned low rank gp, where rank
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
        :param L : A matrix such that :math:`L L^\top \approx K`, where :math:`K` is the
            prior covariance matrix of the GP.
        :type L : array-like or None
        :param Lp : A matrix such that :math:`L_p L_p^\top \approx K_p`, where :math:`K_p` is the
            prior covariance matrix of the GP on the inducing points.
        :type Lp : array-like or None
        :param sigma: Noise standard deviation of the data we condition on. Defaults to 0.
        :type sigma: float
        :param jitter: A small amount to add to the diagonal for stability. Defaults to 1e-6.
        :type jitter: float
        :param y_cov_factor: A matrix :math:`\Sigma_L` such that
            :math:`\Sigma_L\cdot\Sigma_L` is the covaraince of `y`.
            Only required if `with_uncertainty=True`. Defaults to None.
        :type y_cov_factor: array-like
        :param y_is_mean: Wether to consider y the GP mean or a noise measurment
            subject to `sigma` or `y_cov_factor`. Has no effect if `L` is passed.
            Defaults to False.
        :type y_is_mean: bool
        :param with_uncertainty: Wether to compute predictive uncertainty and
            intermediate covariance functions. Defaults to False.
        :type with_uncertainty: bool
        :return: conditional_mean - The conditioned Gaussian process mean function.
        :rtype: function
        """
        x = ensure_2d(x)
        xu = ensure_2d(xu)
        Kuf = cov_func(xu, x)

        if Lp is None:
            Lp = _get_L(xu, cov_func, jitter)

        A = solve_triangular(Lp, Kuf, lower=True)

        r = y - mu
        if y_is_mean:
            r_l, A_l = r, A
        else:
            r_l, A_l = _process_sigma(sigma, r, A, jitter=jitter)
        LBB = stabilize(dot(A_l, A.T), 1)
        L_B = cholesky(LBB)

        c = solve_triangular(L_B, dot(A, r_l), lower=True)
        weights = solve_triangular(Lp.T, solve_triangular(L_B.T, c))

        self.cov_func = cov_func
        self.landmarks = xu
        self.weights = weights
        self.mu = mu
        self.jitter = jitter
        self.n_input_features = xu.shape[1]
        self.n_obs = x.shape[0]

        self._state_variables = {"landmarks", "weights", "mu", "jitter"}

        if not with_uncertainty:
            return

        self.L = Lp
        self._state_variables.add("L")

        Cs = dot(Lp, L_B)
        self.Cs = Cs
        self._state_variables.add("Cs")

        if not y_is_mean:
            return

        y_l = y_cov_factor
        C = solve_triangular(L_B, dot(A, y_l), lower=True)
        Z = solve_triangular(L_B.T, C)
        W = solve_triangular(Lp.T, Z)
        self.W = W
        self._state_variables.add("W")

    def _mean(self, Xnew):
        cov_func = self.cov_func
        xu = self.landmarks
        weights = self.weights
        mu = self.mu

        Kus = cov_func(Xnew, xu)
        return mu + dot(Kus, weights)

    def _covariance(self, Xnew, diag=False):
        _check_covariance(self)
        cov_func = self.cov_func
        xu = self.landmarks
        L = self.L
        Cs = self.Cs

        Kus = cov_func(xu, Xnew)
        C = solve_triangular(Cs, Kus, lower=True)
        As = solve_triangular(L, Kus, lower=True)

        if diag:
            Kss = cov_func.diag(Xnew)
            var = Kss - arraysum(square(As), axis=0) + arraysum(square(C), axis=0)
            return var
        else:
            cov = cov_func(Xnew, Xnew) - dot(As.T, As) + dot(C.T, C)
            return cov

    def _mean_covariance(self, Xnew, diag=True):
        _check_uncertainty(self)
        cov_func = self.cov_func
        xu = self.landmarks
        W = self.W

        Kus = cov_func(Xnew, xu)
        cov_L = Kus @ W

        if diag:
            var = arraysum(cov_L * cov_L, axis=1)
            return var
        else:
            cov = cov_L @ cov_L.T
            return cov


class LandmarksConditional(_LandmarksConditional, Predictor):
    pass


class ExpLandmarksConditional(_LandmarksConditional, ExpPredictor):
    pass


class LandmarksConditionalTime(_LandmarksConditional, PredictorTime):
    pass


class _LandmarksConditionalCholesky:
    def __init__(
        self,
        xu,
        pre_transformation,
        mu,
        cov_func,
        n_obs,
        L=None,
        sigma=DEFAULT_SIGMA,
        jitter=DEFAULT_JITTER,
        y_is_mean=False,
        with_uncertainty=False,
    ):
        """
        The mean function of the conditioned low rank gp, where rank
        is the number of landmark points.

        :param xu: The landmark points.
        :type xu: array-like
        :param pre_transformation: The pre transform latent function representation.
        :type pre_transformation: array-like
        :param mu: The original Gaussian process mean.
        :type mu: float
        :param cov_func: The Gaussian process covariance function.
        :type cov_func: function
        :param n_obs: The number of observations/cells trained on. Used for normalization.
        :type n_obs: int
        :param L : A matrix such that :math:`L L^\top \approx K`, where :math:`K` is the
            covariance matrix of the Gaussian Process.
        :type L : array-like or None
        :param sigma: Standard deviation of `pre_transformation`. Defaults to 0.
        :type sigma: float
        :param jitter: A small amount to add to the diagonal for stability. Defaults to 1e-6.
        :type jitter: float
        :param y_is_mean: Wether to consider y the GP mean or a noise measurment
            subject to `sigma`. Has no effect if `L` is passed.
            Defaults to False.
        :type y_is_mean: bool
        :param with_uncertainty: Wether to compute predictive uncertainty and
            intermediate covariance functions. Defaults to False.
        :type with_uncertainty: bool
        :return: conditional_mean - The conditioned Gaussian process mean function.
        :rtype: function
        """
        xu = ensure_2d(xu)
        if L is None:
            logger.info("Recomputing covariance decomposition for predictive function.")
            if y_is_mean:
                logger.debug("Assuming y is the mean of the GP.")
                L = _get_L(xu, cov_func, jitter)
            else:
                logger.debug("Assuming y is not the mean of the GP.")
                y_cov_factor = _sigma_to_y_cov_factor(sigma, None, xu.shape[0])
                sigma = None
                L = _get_L(xu, cov_func, jitter, y_cov_factor)

        weights = solve_triangular(L.T, pre_transformation)

        self.cov_func = cov_func
        self.landmarks = xu
        self.weights = weights
        self.mu = mu
        self.jitter = jitter
        self.n_input_features = xu.shape[1]
        self.n_obs = n_obs

        self._state_variables = {"landmarks", "weights", "mu", "jitter"}

        if not with_uncertainty:
            return

        self.L = L
        self._state_variables.add("L")

        try:
            Stds = diagonal(sigma)
        except ValueError:
            # sigma seems to be scalar
            Stds = eye(xu.shape[0]) * sigma

        W = solve_triangular(L.T, Stds)
        self.W = W
        self._state_variables.add("W")

    def _mean(self, Xnew):
        cov_func = self.cov_func
        xu = self.landmarks
        weights = self.weights
        mu = self.mu

        Kus = cov_func(Xnew, xu)
        return mu + dot(Kus, weights)

    def _covariance(self, Xnew, diag=True):
        _check_covariance(self)

        cov_func = self.cov_func
        landmarks = self.landmarks
        L = self.L

        K = cov_func(landmarks, Xnew)
        A = solve_triangular(L, K, lower=True)

        if diag:
            Kss = cov_func.diag(Xnew)
            var = Kss - arraysum(square(A), axis=0)
            return var
        else:
            cov = cov_func(Xnew, Xnew) - dot(A.T, A)
            return cov

    def _mean_covariance(self, Xnew, diag=True):
        _check_uncertainty(self)

        cov_func = self.cov_func
        xu = self.landmarks
        W = self.W

        Kus = cov_func(Xnew, xu)
        cov_L = Kus @ W

        if diag:
            var = arraysum(cov_L * cov_L, axis=1)
            return var
        else:
            cov = cov_L @ cov_L.T
            return cov


class LandmarksConditionalCholesky(_LandmarksConditionalCholesky, Predictor):
    pass


class ExpLandmarksConditionalCholesky(_LandmarksConditionalCholesky, ExpPredictor):
    pass


class LandmarksConditionalCholeskyTime(_LandmarksConditionalCholesky, PredictorTime):
    pass
