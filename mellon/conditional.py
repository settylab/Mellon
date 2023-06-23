from jax.numpy import dot, square, isnan, any
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import solve_triangular
from .util import ensure_2d, stabilize, DEFAULT_JITTER, Log
from .base_predictor import Predictor, PredictorTime


logger = Log()


class _FullConditionalMean:
    def __init__(
        self,
        x,
        y,
        mu,
        cov_func,
        sigma=0,
        jitter=DEFAULT_JITTER,
    ):
        """
        The mean function of the conditioned Gaussian process.

        :param x: The training instances.
        :type x: array-like
        :param y: The function value at each point in x.
        :type y: array-like
        :param mu: The original Gaussian process mean.
        :type mu: float
        :param cov_func: The Gaussian process covariance function.
        :type cov_func: function
        :param sigma: White moise standard deviation. Defaults to 0.
        :type sigma: float
        :param jitter: A small amount to add to the diagonal for stability. Defaults to 1e-6.
        :type jitter: float
        :return: conditional_mean - The conditioned Gaussian process mean function.
        :rtype: function
        """
        x = ensure_2d(x)
        sigma2 = square(sigma)
        K = cov_func(x, x)
        sigma2 = max(sigma2, jitter)
        L = cholesky(stabilize(K, jitter=sigma2))
        if any(isnan(L)):
            message = (
                f"Covariance not positively definite with jitter={jitter}. "
                "Consider increasing the jitter for numerical stabilization."
            )
            logger.error(message)
            raise ValueError(message)
        r = y - mu
        weights = solve_triangular(L.T, solve_triangular(L, r, lower=True))

        self.cov_func = cov_func
        self.x = x
        self.weights = weights
        self.mu = mu
        self.n_input_features = x.shape[1]

    def _data_dict(self):
        return {
            "x": self.x,
            "weights": self.weights,
            "mu": self.mu,
        }

    def _predict(self, Xnew):
        cov_func = self.cov_func
        x = self.x
        weights = self.weights
        mu = self.mu

        Kus = cov_func(Xnew, x)
        return mu + dot(Kus, weights)


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
        sigma=0,
        jitter=DEFAULT_JITTER,
    ):
        """
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
        :param sigma: White moise standard deviation. Defaults to 0.
        :type sigma: float
        :param jitter: A small amount to add to the diagonal for stability. Defaults to 1e-6.
        :type jitter: float
        :return: conditional_mean - The conditioned Gaussian process mean function.
        :rtype: function
        """
        x = ensure_2d(x)
        xu = ensure_2d(xu)
        sigma2 = square(sigma)
        Kuu = cov_func(xu, xu)
        Kuf = cov_func(xu, x)
        Luu = cholesky(stabilize(Kuu, jitter))
        if any(isnan(Luu)):
            message = (
                f"Covariance of landmarks not positively definite with jitter={jitter}. "
                "Consider increasing the jitter for numerical stabilization."
            )
            logger.error(message)
            raise ValueError(message)
        A = solve_triangular(Luu, Kuf, lower=True)
        sigma2 = max(sigma2, jitter)
        L_B = cholesky(stabilize(dot(A, A.T), sigma2))
        r = y - mu
        c = solve_triangular(L_B, dot(A, r), lower=True)
        z = solve_triangular(L_B.T, c)
        weights = solve_triangular(Luu.T, z)

        self.cov_func = cov_func
        self.landmarks = xu
        self.weights = weights
        self.mu = mu
        self.n_input_features = xu.shape[1]

    def _data_dict(self):
        return {
            "landmarks": self.landmarks,
            "weights": self.weights,
            "mu": self.mu,
        }

    def _predict(self, Xnew):
        cov_func = self.cov_func
        xu = self.landmarks
        weights = self.weights
        mu = self.mu

        Kus = cov_func(Xnew, xu)
        return mu + dot(Kus, weights)


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
        sigma=0,
        jitter=DEFAULT_JITTER,
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
        :param sigma: White moise standard deviation. Defaults to 0.
        :type sigma: float
        :param jitter: A small amount to add to the diagonal for stability. Defaults to 1e-6.
        :type jitter: float
        :return: conditional_mean - The conditioned Gaussian process mean function.
        :rtype: function
        """
        xu = ensure_2d(xu)
        sigma2 = square(sigma)
        K = cov_func(xu, xu)
        sigma2 = max(sigma2, jitter)
        L = cholesky(stabilize(K, jitter=sigma2))
        if any(isnan(L)):
            message = (
                f"Covariance not positively definite with jitter={jitter}. "
                "Consider increasing the jitter for numerical stabilization."
            )
            logger.error(message)
            raise ValueError(message)
        weights = solve_triangular(L.T, pre_transformation)

        self.cov_func = cov_func
        self.landmarks = xu
        self.weights = weights
        self.mu = mu
        self.n_input_features = xu.shape[1]

    def _data_dict(self):
        return {
            "landmarks": self.landmarks,
            "weights": self.weights,
            "mu": self.mu,
        }

    def _predict(self, Xnew):
        cov_func = self.cov_func
        xu = self.landmarks
        weights = self.weights
        mu = self.mu

        Kus = cov_func(Xnew, xu)
        return mu + dot(Kus, weights)


class LandmarksConditionalMeanCholesky(_LandmarksConditionalMeanCholesky, Predictor):
    pass


class LandmarksConditionalMeanCholeskyTime(
    _LandmarksConditionalMeanCholesky, PredictorTime
):
    pass
