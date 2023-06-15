from .decomposition import DEFAULT_METHOD
from .base_model import BaseEstimator, DEFAULT_COV_FUNC
from .inference import (
    compute_conditional_mean,
    compute_conditional_mean_y,
    DEFAULT_N_ITER,
    DEFAULT_INIT_LEARN_RATE,
    DEFAULT_OPTIMIZER,
)
from .parameters import (
    compute_landmarks,
    DEFAULT_N_LANDMARKS,
)
from .util import (
    DEFAULT_JITTER,
    Log,
)
from .helper import vector_map
from .validation import (
    _validate_positive_float,
    _validate_float,
    _validate_array,
)


DEFAULT_D_METHOD = "embedding"

logger = Log()


class FunctionEstimator(BaseEstimator):
    R"""
    This class implements a Function Estimator that uses a conditional normal distribution
    to smoothen and extend a function on all cell states using the Mellon abstractions.

    Parameters
    ----------
    cov_func_curry : function or type
        A curry that takes one length scale argument and returns a covariance function
        of the form k(x, y) :math:`\rightarrow` float. Defaults to Matern52.

    n_landmarks : int, optional
        The number of landmark points. If less than 1 or greater than or equal to the
        number of training points, inducing points will not be computed or used. Defaults to 5000.

    jitter : float, optional
        A small amount added to the diagonal of the covariance matrix to ensure numerical stability.
        Defaults to 1e-6.

    landmarks : array-like or None, optional
        Points used to quantize the data for the approximate covariance. If None, landmarks are
        set as k-means centroids with k=n_landmarks. This is ignored if n_landmarks is greater than
        or equal to the number of training points. Defaults to None.

    nn_distances : array-like or None, optional
        The nearest neighbor distances at each data point. If None, computes the nearest neighbor
        distances automatically, with a KDTree if the dimensionality of the data is less than 20,
        or a BallTree otherwise. Defaults to None.

    mu : float
        The mean of the Gaussian process :math:`\mu`. Defaults to 0.

    ls : float or None, optional
        The length scale for the Gaussian process covariance function.
        If None (default), the length scale is automatically selected based on
        a heuristic link between the nearest neighbor distances and the optimal
        length scale.

    ls_factor : float, optional
        A scaling factor applied to the length scale when it's automatically
        selected. It is used to manually adjust the automatically chosen length
        scale for finer control over the model's sensitivity to variations in the data.

    cov_func : function or None, optional
        The Gaussian process covariance function of the form k(x, y) :math:`\rightarrow` float.
        If None, automatically generates the covariance function cov_func = cov_func_curry(ls).
        Defaults to None.

    sigma : float, optional
        The standard deviation of the white noise. Defaults to 0.

    jit : bool, optional
        Use JAX just-in-time compilation for the loss function and its gradient during optimization.
        Defaults to False.
    """

    def __init__(
        self,
        cov_func_curry=DEFAULT_COV_FUNC,
        n_landmarks=DEFAULT_N_LANDMARKS,
        method=DEFAULT_METHOD,
        jitter=DEFAULT_JITTER,
        optimizer=DEFAULT_OPTIMIZER,
        n_iter=DEFAULT_N_ITER,
        init_learn_rate=DEFAULT_INIT_LEARN_RATE,
        landmarks=None,
        nn_distances=None,
        mu=0,
        ls=None,
        ls_factor=1,
        cov_func=None,
        sigma=0,
        jit=True,
    ):
        super().__init__(
            cov_func_curry=cov_func_curry,
            n_landmarks=n_landmarks,
            rank=1.0,
            jitter=jitter,
            landmarks=landmarks,
            nn_distances=nn_distances,
            mu=mu,
            ls=ls,
            ls_factor=ls_factor,
            cov_func=cov_func,
            jit=jit,
        )
        self.mu = _validate_float(mu, "mu")
        self.sigma = _validate_positive_float(sigma, "sigma")

    def __call__(self, x=None, y=None):
        """This calls self.fit_predict(x, y):
        Compute the conditional mean and return the smoothed function values
        at the points x.

        :param x: The training instances to estimate function.
        :type x: array-like
        :param y: The training function values on cell states.
        :type y: array-like
        :return: condition_mean - The conditional mean function value at each test point in x.
        :rtype: array-like
        """
        return self.fit_predict(x=x, y=y)

    def prepare_inference(self, x):
        R"""
        Set all attributes in preparation. It is not necessary to call this
        function before calling fit.

        :param x: The cell states.
        :type x: array-like
        :param y: The function values on the cell states.
        :type y: array-like
        :return: loss_func, initial_value - The Bayesian loss function and
            initial guess for optimization.
        :rtype: function, array-like
        """
        self._set_x(x)
        if self.ls is None:
            self._prepare_attribute("nn_distances")
        self._prepare_attribute("ls")
        self._prepare_attribute("cov_func")
        self._prepare_attribute("landmarks")
        return

    def compute_conditional(self, x=None, y=None):
        R"""
        Compute and return the conditional mean function.

        :param x: The training instances to estimate density function.
        :type x: array-like
        :param y: The training function values on cell states.
        :type y: array-like
        :return: condition_mean_function - The conditional mean function.
        :rtype: array-like
        """
        if x is None:
            x = self.x
        else:
            x = _validate_array(x, "x")
        if self.x is not None and self.x is not x:
            message = (
                "self.x has been set already, but is not equal to the argument x. "
                "Current landmarks might be inapropriate."
            )
            logger.warning(message)
        if self.x is None and x is None:
            message = "Required argument x is missing and self.x has not been set."
            raise ValueError(message)
        if y is None:
            message = "Required argument y is missing."
            raise ValueError(message)
        landmarks = self.landmarks
        mu = self.mu
        cov_func = self.cov_func
        sigma = self.sigma
        jitter = self.jitter
        conditional = compute_conditional_mean(
            x,
            landmarks,
            None,
            y,
            mu,
            cov_func,
            sigma,
            jitter=jitter,
        )
        self.conditional = conditional
        return conditional

    def fit(self, x=None, y=None):
        R"""
        Fit the model from end to end.

        :param x: The training cell states.
        :type x: array-like
        :param y: The training function values on cell states.
        :type y: array-like
        :param build_predict: Whether or not to build the prediction function.
            Defaults to True.
        :type build_predict: bool
        :return: self - A fitted instance of this estimator.
        :rtype: Object
        """

        self.prepare_inference(x)
        self.compute_conditional(x, y)
        return self

    @property
    def predict(self):
        R"""
        An instance of the :class:`mellon.Predictor` that predicts the function values at each point in x.

        It contains a __call__ method which can be used to predict the function values
        The instance also supports serialization features which allows for saving
        and loading the predictor state. Refer to :class:`mellon.Predictor` documentation for more details.

        :param x: The new data to predict.
        :type x: array-like
        :return: condition_mean - The conditional mean function value at each test point in x.
        :rtype: array-like
        """
        return self.conditional

    def fit_predict(self, x=None, y=None):
        R"""
        Compute the conditional mean and return the smoothed function values
        at the points x.

        :param x: The training instances to estimate function.
        :type x: array-like
        :param y: The training function values on cell states.
        :type y: array-like
        :return: condition_mean - The conditional mean function value at each test point in x.
        :rtype: array-like
        """
        x = _validate_array(x, "x")
        y = _validate_array(y, "y")

        self.fit(x, y)
        return self.predict(x)

    def multi_fit_predict(self, x=None, Y=None, Xnew=None):
        R"""
        Compute the conditional mean and return the smoothed function values
        at the points Xnew for each line of values in Y.

        :param x: The training instances to estimate density function.
        :type x: array-like
        :param y: The training function values on cell states.
        :type y: array-like
        :param Xnew: The new data to predict.
        :type Xnew: array-like
        :return: condition_mean - The conditional mean function value at each test point in x.
        :rtype: array-like
        """

        self._set_x(x)
        Y = _validate_array(Y, "Y")
        Xnew = _validate_array(Xnew, "Xnew", optional=True)
        if self.ls is None:
            self._prepare_attribute("nn_distances")
        self._prepare_attribute("ls")
        self._prepare_attribute("cov_func")
        if Xnew is None:
            Xnew = x
            self._prepare_attribute("landmarks")
            landmarks = self.landmarks
        else:
            n_landmarks = self.n_landmarks
            logger.info(f"Computing {n_landmarks:,} landmarks for Xnew.")
            landmarks = compute_landmarks(Xnew, n_landmarks)

        mu = self.mu
        cov_func = self.cov_func
        sigma = self.sigma
        jitter = self.jitter
        jit = self.jit

        conditional = compute_conditional_mean_y(
            x,
            landmarks,
            Xnew,
            mu,
            cov_func,
            sigma,
            jitter=jitter,
        )

        return vector_map(conditional, Y, do_jit=jit).squeeze()
