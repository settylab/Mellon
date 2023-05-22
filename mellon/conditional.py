from jax.numpy import dot, square, isnan, any, atleast_2d
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import solve_triangular
from .util import stabilize, DEFAULT_JITTER, Log, make_serializable
from .base_predictor import Predictor


logger = Log()


def make_2d(X):
    return atleast_2d(X.T).T


class FullConditionalMean(Predictor):
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
        x = make_2d(x)
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

    def _data_dict(self):
        return {
            "x": make_serializable(self.x),
            "weights": make_serializable(self.weights),
            "mu": make_serializable(self.mu),
        }

    def __call__(self, Xnew):
        cov_func = self.cov_func
        x = self.x
        weights = self.weights
        mu = self.mu

        Xnew = make_2d(Xnew)
        Kus = cov_func(Xnew, x)
        return mu + dot(Kus, weights)


class FullConditionalMeanY(Predictor):
    def __init__(
        self,
        x,
        Xnew,
        mu,
        cov_func,
        sigma=0,
        jitter=DEFAULT_JITTER,
    ):
        """
        The mean function of the conditioned Gaussian process for fixed
        output locations Xnew and therefor flexible output values y.

        :param x: The training instances.
        :type x: array-like
        :param Xnew: The output locations.
        :type Xnew: array-like
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
        x = make_2d(x)
        Xnew = make_2d(Xnew)
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
        Kus = cov_func(Xnew, x)

        self.cov_func = cov_func
        self.L = L
        self.Kus = Kus
        self.mu = mu

    def _data_dict(self):
        return {
            "L": make_serializable(self.L),
            "Kuus": make_serializable(self.Kuus),
            "mu": make_serializable(self.mu),
        }

    def __call__(self, y):
        L = self.L
        Kus = self.Kus
        mu = self.mu

        weights = solve_triangular(L.T, solve_triangular(L, y, lower=True))
        return mu + dot(Kus, weights)


class LandmarksConditionalMean(Predictor):
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
        x = make_2d(x)
        xu = make_2d(xu)
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

    def _data_dict(self):
        return {
            "landmarks": make_serializable(self.landmarks),
            "weights": make_serializable(self.weights),
            "mu": make_serializable(self.mu),
        }

    def __call__(self, Xnew):
        cov_func = self.cov_func
        xu = self.landmarks
        weights = self.weights
        mu = self.mu

        Xnew = make_2d(Xnew)
        Kus = cov_func(Xnew, xu)
        return mu + dot(Kus, weights)


class LandmarksConditionalMeanCholesky(Predictor):
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
        xu = make_2d(xu)
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

    def _data_dict(self):
        return {
            "landmarks": make_serializable(self.landmarks),
            "weights": make_serializable(self.weights),
            "mu": make_serializable(self.mu),
        }

    def __call__(self, Xnew):
        cov_func = self.cov_func
        xu = self.landmarks
        weights = self.weights
        mu = self.mu

        Xnew = make_2d(Xnew)
        Kus = cov_func(Xnew, xu)
        return mu + dot(Kus, weights)


class LandmarksConditionalMeanY(Predictor):
    def __init__(
        self,
        x,
        xu,
        Xnew,
        mu,
        cov_func,
        sigma=0,
        jitter=DEFAULT_JITTER,
    ):
        """
        The mean function of the conditioned low rank gp, where rank
        is less than the number of landmark points, and for fixed
        output locations Xnew and therefor flexible output values y.

        :param x: The training instances.
        :type x: array-like
        :param xu: The landmark points.
        :type xu: array-like
        :param Xnew: The output locations.
        :type Xnew: array-like
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
        x = make_2d(x)
        xu = make_2d(xu)
        Xnew = make_2d(Xnew)
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
        Kus = cov_func(Xnew, xu)

        self.cov_func = cov_func
        self.L_B = L_B
        self.A = A
        self.Luu = Luu
        self.Kus = Kus
        self.mu = mu

    def _data_dict(self):
        return {
            "L_B": make_serializable(self.L_B),
            "A": make_serializable(self.A),
            "Luu": make_serializable(self.Luu),
            "Kus": make_serializable(self.Kus),
            "mu": make_serializable(self.mu),
        }

    def __call__(self, y):
        L_B = self.L_B
        A = self.A
        Luu = self.Luu
        Kus = self.Kus
        mu = self.mu

        r = y - mu
        c = solve_triangular(L_B, dot(A, r), lower=True)
        z = solve_triangular(L_B.T, c)
        weights = solve_triangular(Luu.T, z)
        return mu + dot(Kus, weights)
