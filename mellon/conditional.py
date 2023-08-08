from jax.numpy import dot, square, isnan, any, eye
from jax.numpy import sum as arraysum
from jax.numpy import diag as diagonal
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import solve_triangular
from .util import ensure_2d, stabilize, DEFAULT_JITTER, Log
from .base_predictor import Predictor, PredictorTime
from .decomposition import DEFAULT_SIGMA


logger = Log()


def _get_L(x, cov_func, jitter=DEFAULT_JITTER):
    K = cov_func(x, x)
    L = cholesky(stabilize(K, jitter=jitter))
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


class _FullConditionalMean:
    def __init__(
        self,
        x,
        y,
        mu,
        cov_func,
        sigma=DEFAULT_SIGMA,
        jitter=DEFAULT_JITTER,
        y_cov_factor=None,
        with_uncertainty=False,
    ):
        R"""
        The mean function of the conditioned Gaussian process.

        :param x: The training instances.
        :type x: array-like
        :param y: The function value at each point in x.
        :type y: array-like
        :param mu: The original Gaussian process mean.
        :type mu: float
        :param cov_func: The Gaussian process covariance function.
        :type cov_func: function
        :param sigma: Noise standard deviation of the data we condition on. Defaults to 0.
        :type sigma: float
        :param jitter: A small amount to add to the diagonal for stability. Defaults to 1e-6.
        :type jitter: float
        :param y_cov_factor: A matrix :math:`\Sigma_L` such that
            :math:`\Sigma_L\cdot\Sigma_L` is the covaraince of `y`.
            Only required if `with_uncertainty=True`. Defaults to None.
        :type y_cov_factor: array-like
        :param with_uncertainty: Wether to compute covariance functions and
            predictive uncertainty. Defaults to False.
        :type with_uncertainty: bool
        :return: conditional_mean - The conditioned Gaussian process mean function.
        :rtype: function
        """
        x = ensure_2d(x)

        L = _get_L(x, cov_func, jitter)
        r = y - mu
        weights = solve_triangular(L.T, solve_triangular(L, r, lower=True))

        self.cov_func = cov_func
        self.x = x
        self.weights = weights
        self.mu = mu
        self.jitter = jitter
        self.sigma = sigma
        self.n_input_features = x.shape[1]

        self._state_variables = {"x", "weights", "mu", "sigma", "jitter"}

        if not with_uncertainty:
            return

        self.L = L
        self._state_variables.add("L")

        if y_cov_factor is not None and sigma > 0:
            raise ValueError(
                "One can specify either `sigma` or `y_cov_factor` to describe input noise, but not both."
            )

        if y_cov_factor is None:
            try:
                y_cov_factor = diagonal(sigma)
            except ValueError:
                y_cov_factor = eye(x.shape[0]) * sigma

        W = solve_triangular(L.T, solve_triangular(L, y_cov_factor, lower=True))
        self.W = W
        self._state_variables.add("W")

    def _predict(self, Xnew):
        cov_func = self.cov_func
        x = self.x
        weights = self.weights
        mu = self.mu

        Kus = cov_func(Xnew, x)
        return mu + dot(Kus, weights)

    def covariance(self, Xnew, diag=True):
        """
        Computes the covariance of the Gaussian Process distribution of functions
        over new data points or cell states.

        Parameters:
        Xnew : array-like, shape (n_samples, n_features)
            The new data points for which to compute the covariance.
        diag : boolean, optional (default=True)
            Whether to return the variance (True) or the full covariance matrix (False).

        Returns:
        var : array-like, shape (n_samples,)
            If diag=True, returns the variances for each sample.
        cov : array-like, shape (n_samples, n_samples)
            If diag=False, returns the full covariance matrix between samples.
        """
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

    def mean_covariance(self, Xnew, diag=True):
        """
        Computes the uncertainty of the mean of the Gaussian process induced by
        the uncertainty of the latent representation of the mean function.

        Parameters:
        Xnew : array-like, shape (n_samples, n_features)
            The new data points for which to compute the uncertainty.
        diag : boolean, optional (default=True)
            Whether to compute the variance (True) or the full covariance matrix (False).

        Returns:
        var : array-like, shape (n_samples,)
            If diag=True, returns the variances for each sample.
        cov : array-like, shape (n_samples, n_samples)
            If diag=False, returns the full covariance matrix between samples.
        """
        _check_uncertainty(self)
        cov_func = self.cov_func
        x = self.x
        W = self.W

        Kus = cov_func(Xnew, x)
        cov_L = dot(Kus, W)

        if diag:
            var = arraysum(cov_L * cov_L, axis=1)
            return var
        else:
            cov = dot(cov_L, cov_L.T)
            return cov


class FullConditionalMean(_FullConditionalMean, Predictor):
    pass


class FullConditionalMeanTime(_FullConditionalMean, PredictorTime):
    pass


class _LandmarksConditionalMean:
    def __init__(
        self,
        x,
        xu,
        y,
        mu,
        cov_func,
        sigma=DEFAULT_SIGMA,
        jitter=DEFAULT_JITTER,
        y_cov_factor=None,
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
        :param sigma: Noise standard deviation of the data we condition on. Defaults to 0.
        :type sigma: float
        :param jitter: A small amount to add to the diagonal for stability. Defaults to 1e-6.
        :type jitter: float
        :param y_cov_factor: A matrix :math:`\Sigma_L` such that
            :math:`\Sigma_L\cdot\Sigma_L` is the covaraince of `y`.
            Only required if `with_uncertainty=True`. Defaults to None.
        :type y_cov_factor: array-like
        :param with_uncertainty: Wether to compute predictive uncertainty and
            intermediate covariance functions. Defaults to False.
        :type with_uncertainty: bool
        :return: conditional_mean - The conditioned Gaussian process mean function.
        :rtype: function
        """
        x = ensure_2d(x)
        xu = ensure_2d(xu)
        Kuf = cov_func(xu, x)
        L = _get_L(xu, cov_func, jitter)
        A = solve_triangular(L, Kuf, lower=True)

        L_B = cholesky(stabilize(dot(A, A.T), jitter))
        r = y - mu
        c = solve_triangular(L_B, dot(A, r), lower=True)
        z = solve_triangular(L_B.T, c)
        weights = solve_triangular(L.T, z)

        self.cov_func = cov_func
        self.landmarks = xu
        self.weights = weights
        self.mu = mu
        self.sigma = sigma
        self.jitter = jitter
        self.n_input_features = xu.shape[1]

        self._state_variables = {"landmarks", "weights", "mu", "sigma", "jitter"}

        if not with_uncertainty:
            return

        self.L = L
        self._state_variables.add("L")

        if y_cov_factor is not None and any(sigma > 0):
            raise ValueError(
                "One can specify either `sigma` or `y_cov_factor` to describe input noise, but not both."
            )
        if y_cov_factor is None:
            try:
                y_cov_factor = diagonal(sigma)
            except ValueError:
                y_cov_factor = eye(xu.shape[0]) * sigma

        C = solve_triangular(L_B, dot(A, y_cov_factor), lower=True)
        Z = solve_triangular(L_B.T, C)
        W = solve_triangular(L.T, Z)
        self.W = W
        self._state_variables.add("W")

    def _predict(self, Xnew):
        cov_func = self.cov_func
        xu = self.landmarks
        weights = self.weights
        mu = self.mu

        Kus = cov_func(Xnew, xu)
        return mu + dot(Kus, weights)

    def covariance(self, Xnew, diag=False):
        """
        Computes the covariance of the Gaussian Process distribution of functions
        over new data points or cell states.

        Parameters:
        Xnew : array-like, shape (n_samples, n_features)
            The new data points for which to compute the covariance.
        diag : boolean, optional (default=True)
            Whether to return the variance (True) or the full covariance matrix (False).

        Returns:
        var : array-like, shape (n_samples,)
            If diag=True, returns the variances for each sample.
        cov : array-like, shape (n_samples, n_samples)
            If diag=False, returns the full covariance matrix between samples.
        """
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

    def mean_covariance(self, Xnew, diag=True):
        """
        Computes the uncertainty of the mean of the Gaussian process induced by
        the uncertainty of the latent representation of the mean function.

        Parameters:
        Xnew : array-like, shape (n_samples, n_features)
            The new data points for which to compute the uncertainty.
        diag : boolean, optional (default=True)
            Whether to compute the variance (True) or the full covariance matrix (False).

        Returns:
        var : array-like, shape (n_samples,)
            If diag=True, returns the variances for each sample.
        cov : array-like, shape (n_samples, n_samples)
            If diag=False, returns the full covariance matrix between samples.
        """
        _check_uncertainty(self)
        cov_func = self.cov_func
        xu = self.landmarks
        W = self.W

        Kus = cov_func(Xnew, xu)
        cov_L = dot(Kus, W)

        if diag:
            var = arraysum(cov_L * cov_L, axis=1)
            return var
        else:
            cov = dot(cov_L, cov_L.T)
            return cov


class LandmarksConditionalMean(_LandmarksConditionalMean, Predictor):
    pass


class LandmarksConditionalMeanTime(_LandmarksConditionalMean, PredictorTime):
    pass


class _LandmarksConditionalMeanCholesky:
    def __init__(
        self,
        xu,
        pre_transformation,
        mu,
        cov_func,
        sigma=DEFAULT_SIGMA,
        jitter=DEFAULT_JITTER,
        pre_transformation_std=None,
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
        :param sigma: Noise standard deviation of the data we condition on. Defaults to 0.
        :type sigma: float
        :param jitter: A small amount to add to the diagonal for stability. Defaults to 1e-6.
        :type jitter: float
        :param pre_transformation_std: Standard deviation of `pre_transformation`.
            Only required if `with_uncertainty=True`. Defaults to None.
        :type pre_transformation_std: array-like
        :param with_uncertainty: Wether to compute predictive uncertainty and
            intermediate covariance functions. Defaults to False.
        :type with_uncertainty: bool
        :return: conditional_mean - The conditioned Gaussian process mean function.
        :rtype: function
        """
        xu = ensure_2d(xu)
        L = _get_L(xu, cov_func, jitter)
        weights = solve_triangular(L.T, pre_transformation)

        self.cov_func = cov_func
        self.landmarks = xu
        self.weights = weights
        self.mu = mu
        self.sigma = sigma
        self.jitter = jitter
        self.n_input_features = xu.shape[1]

        self._state_variables = {"landmarks", "weights", "mu", "sigma", "jitter"}

        if not with_uncertainty:
            return

        self.L = L
        self._state_variables.add("L")

        if pre_transformation_std is not None and sigma > 0:
            raise ValueError(
                "One can specify either `sigma` or `pre_transformation_std` "
                "to describe input noise, but not both."
            )
        if pre_transformation_std is None:
            logger.warning(
                "`sigma` is interpreted as standard deviation of `pre_transform`."
            )
            try:
                Stds = diagonal(sigma)
            except ValueError:
                # sigma seems to be scalar
                Stds = eye(xu.shape[0]) * sigma
        else:
            Stds = diagonal(pre_transformation_std)

        W = solve_triangular(L.T, Stds)
        self.W = W
        self._state_variables.add("W")

    def _predict(self, Xnew):
        cov_func = self.cov_func
        xu = self.landmarks
        weights = self.weights
        mu = self.mu

        Kus = cov_func(Xnew, xu)
        return mu + dot(Kus, weights)

    def covariance(self, Xnew, diag=True):
        """
        Computes the covariance of the Gaussian Process distribution of functions
        over new data points or cell states.

        Parameters:
        Xnew : array-like, shape (n_samples, n_features)
            The new data points for which to compute the covariance.
        diag : boolean, optional (default=True)
            Whether to return the variance (True) or the full covariance matrix (False).

        Returns:
        var : array-like, shape (n_samples,)
            If diag=True, returns the variances for each sample.
        cov : array-like, shape (n_samples, n_samples)
            If diag=False, returns the full covariance matrix between samples.
        """
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

    def mean_covariance(self, Xnew, diag=True):
        """
        Computes the uncertainty of the mean of the Gaussian process induced by
        the uncertainty of the latent representation of the mean function.

        Parameters:
        Xnew : array-like, shape (n_samples, n_features)
            The new data points for which to compute the uncertainty.
        diag : boolean, optional (default=True)
            Whether to compute the variance (True) or the full covariance matrix (False).

        Returns:
        var : array-like, shape (n_samples,)
            If diag=True, returns the variances for each sample.
        cov : array-like, shape (n_samples, n_samples)
            If diag=False, returns the full covariance matrix between samples.
        """
        _check_uncertainty(self)
        cov_func = self.cov_func
        xu = self.landmarks
        W = self.W

        Kus = cov_func(Xnew, xu)
        cov_L = dot(Kus, W)

        if diag:
            var = arraysum(cov_L * cov_L, axis=1)
            return var
        else:
            cov = dot(cov_L, cov_L.T)
            return cov


class LandmarksConditionalMeanCholesky(_LandmarksConditionalMeanCholesky, Predictor):
    pass


class LandmarksConditionalMeanCholeskyTime(
    _LandmarksConditionalMeanCholesky, PredictorTime
):
    pass
