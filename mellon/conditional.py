from jax.numpy import dot, square, isnan, any, isscalar, full
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import solve_triangular
from .util import stabilize, DEFAULT_JITTER, Log
from .helper import ensure_2d, make_serializable
from .base_predictor import Predictor
from .validation import _validate_array, _validate_time_x


logger = Log()


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

    def _data_dict(self):
        return {
            "x": make_serializable(self.x),
            "weights": make_serializable(self.weights),
            "mu": make_serializable(self.mu),
        }

    def __call__(self, Xnew):
        Xnew = _validate_array(Xnew, "Xnew")
        Xnew = ensure_2d(Xnew)

        cov_func = self.cov_func
        x = self.x
        weights = self.weights
        mu = self.mu

        Kus = cov_func(Xnew, x)
        return mu + dot(Kus, weights)


class FullConditionalMeanTimes(FullConditionalMean):
    def __call__(self, Xnew, times=None):
        """
        Call method to use the class instance as a function. This method
        deals with an optional 'times' argument.
        If 'times' is a scalar, it converts it to a 1D array of the same size as 'Xnew'.

        Parameters
        ----------
        Xnew : array-like
            The new data points for prediction.
        times : scalar or array-like, optional
            The time points associated with each cell/row in 'Xnew'.
            If 'times' is a scalar, it will be converted into a 1D array of the same size as 'Xnew'.

        Returns
        -------
        array-like
            Predictions for 'Xnew'.

        Raises
        ------
        ValueError
            If 'times' is an array and its size does not match 'Xnew'.
        """

        # if times is a scalar, convert it into a 1D array of the same size as Xnew
        if isscalar(times):
            times = full(Xnew.shape[0], times)
        Xnew = _validate_time_x(Xnew, times)

        return super().__call__(Xnew)


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
        x = ensure_2d(x)
        Xnew = ensure_2d(Xnew)
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
        y = _validate_array(y, "y")

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

    def _data_dict(self):
        return {
            "landmarks": make_serializable(self.landmarks),
            "weights": make_serializable(self.weights),
            "mu": make_serializable(self.mu),
        }

    def __call__(self, Xnew):
        Xnew = _validate_array(Xnew, "Xnew")
        Xnew = ensure_2d(Xnew)

        cov_func = self.cov_func
        xu = self.landmarks
        weights = self.weights
        mu = self.mu

        Kus = cov_func(Xnew, xu)
        return mu + dot(Kus, weights)


class LandmarksConditionalMeanTimes(LandmarksConditionalMean):
    def __call__(self, Xnew, times=None):
        """
        Call method to use the class instance as a function. This method
        deals with an optional 'times' argument.
        If 'times' is a scalar, it converts it to a 1D array of the same size as 'Xnew'.

        Parameters
        ----------
        Xnew : array-like
            The new data points for prediction.
        times : scalar or array-like, optional
            The time points associated with each cell/row in 'Xnew'.
            If 'times' is a scalar, it will be converted into a 1D array of the same size as 'Xnew'.

        Returns
        -------
        array-like
            Predictions for 'Xnew'.

        Raises
        ------
        ValueError
            If 'times' is an array and its size does not match 'Xnew'.
        """

        # if times is a scalar, convert it into a 1D array of the same size as Xnew
        if isscalar(times):
            times = full(Xnew.shape[0], times)
        Xnew = _validate_time_x(Xnew, times)

        return super().__call__(Xnew)


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

    def _data_dict(self):
        return {
            "landmarks": make_serializable(self.landmarks),
            "weights": make_serializable(self.weights),
            "mu": make_serializable(self.mu),
        }

    def __call__(self, Xnew):
        Xnew = _validate_array(Xnew, "Xnew")
        Xnew = ensure_2d(Xnew)

        cov_func = self.cov_func
        xu = self.landmarks
        weights = self.weights
        mu = self.mu

        Kus = cov_func(Xnew, xu)
        return mu + dot(Kus, weights)


class LandmarksConditionalMeanCholeskyTimes(LandmarksConditionalMeanCholesky):
    def __call__(self, Xnew, times=None):
        """
        Call method to use the class instance as a function. This method
        deals with an optional 'times' argument.
        If 'times' is a scalar, it converts it to a 1D array of the same size as 'Xnew'.

        Parameters
        ----------
        Xnew : array-like
            The new data points for prediction.
        times : scalar or array-like, optional
            The time points associated with each cell/row in 'Xnew'.
            If 'times' is a scalar, it will be converted into a 1D array of the same size as 'Xnew'.

        Returns
        -------
        array-like
            Predictions for 'Xnew'.

        Raises
        ------
        ValueError
            If 'times' is an array and its size does not match 'Xnew'.
        """

        # if times is a scalar, convert it into a 1D array of the same size as Xnew
        if isscalar(times):
            times = full(Xnew.shape[0], times)
        Xnew = _validate_time_x(Xnew, times)

        return super().__call__(Xnew)


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
        x = ensure_2d(x)
        xu = ensure_2d(xu)
        Xnew = ensure_2d(Xnew)
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
        y = _validate_array(y, "y")

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
