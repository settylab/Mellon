from .decomposition import DEFAULT_METHOD
from .base_model import BaseEstimator, DEFAULT_COV_FUNC
from .inference import (
    compute_conditional_mean,
    DEFAULT_N_ITER,
    DEFAULT_INIT_LEARN_RATE,
    DEFAULT_OPTIMIZER,
)
from .parameters import (
    DEFAULT_N_LANDMARKS,
)
from .util import (
    DEFAULT_JITTER,
    Log,
)
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
        x = self.set_x(x)
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
        """
        Trains the model using the provided training data and function values. This includes preparing
        the model for inference and computing the conditional distribution for the given data.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features), default=None
            The training instances where `n_samples` is the number of samples and `n_features` is the
            number of features. Each sample is an array of features representing a point in the feature space.

        y : array-like of shape (n_samples, n_output_features), default=None
            The function values of the training instances. `n_samples` is the number of samples and
            `n_output_features` is the number of function values at each sample.

        Raises
        ------
        ValueError
            If the number of samples in `x` and `y` doesn't match.

        Returns
        -------
        self : object
            This method returns self for chaining.
        """
        x = self.set_x(x)
        y = _validate_array(y, "y")

        n_samples = x.shape[0]
        # Check if the number of samples in x and y match
        if y.shape[0] != n_samples:
            raise ValueError(
                f"X.shape[0] = {n_samples} (n_samples) should equal "
                "y.shape[0] = {y.shape[0]}."
            )

        self.prepare_inference(x)
        self.compute_conditional(x, y)
        return self

    @property
    def predict(self):
        """
        A property that returns an instance of the :class:`mellon.Predictor` class. This predictor can be used
        to predict the function values for new data points by calling the instance like a function.

        The predictor instance also supports serialization features, allowing for saving and loading the
        predictor's state. For more details, refer to the :class:`mellon.Predictor` documentation.

        Returns
        -------
        mellon.Predictor
            A predictor instance that computes the conditional mean function value at each new data point.

        Example
        -------

        >>> y_pred = model.predict(Xnew)

        """
        return self.conditional

    def fit_predict(self, x=None, y=None, Xnew=None):
        """
        Trains the model using the provided training data and function values, then makes predictions
        for new data points. The function computes the conditional mean and returns the smoothed function
        values at the points `Xnew` for each column of values in `y`.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features), default=None
            The training instances where `n_samples` is the number of samples and `n_features` is the number
            of features. Each sample is an array of features representing a point in the feature space.

        y : array-like of shape (n_samples, n_output_features), default=None
            The function values of the training instances. `n_samples` is the number of samples and
            `n_output_features` is the number of function values at each sample.

        Xnew : array-like of shape (n_predict_samples, n_features), default=None
            The new data points to make predictions on where `n_predict_samples` is the number of samples
            to predict and `n_features` is the number of features. If not provided, the predictions will
            be made on the training instances `x`.

        Returns
        -------
        array-like of shape (n_predict_samples, n_output_features)
            The conditional mean function value at each new data point in `Xnew`. The number of predicted
            function values at each sample will match the number of output features in `y`.

        Raises
        ------
        ValueError
            If the number of samples in `x` and `y` don't match, or if the number of features in `x` and
            `Xnew` don't match.
        """
        x = self.set_x(x)
        y = _validate_array(y, "y")
        Xnew = _validate_array(Xnew, "Xnew", optional=True)

        # If Xnew is not provided, default to x
        if Xnew is None:
            Xnew = x
        else:
            # Ensure the number of dimensions in x and Xnew are the same
            if x.ndim != Xnew.ndim:
                raise ValueError(
                    f"The provided arrays, 'x' and 'Xnew', do not have the same number of dimensions. "
                    f"'x' is {x.ndim}-D and 'Xnew' is {Xnew.ndim}-D. Please provide arrays with consistent dimensionality."
                )

            # If both arrays are multi-dimensional, ensure they have the same number of features
            if x.ndim > 1 and x.shape[1] != Xnew.shape[1]:
                raise ValueError(
                    f"The provided arrays, 'x' and 'Xnew', should have the same number of features. "
                    f"Got Xnew.shape[1] = {Xnew.shape[1]}, but expected it to be equal to x.shape[1] = {x.shape[1]}. "
                    "Please provide arrays with the same number of features."
                )

        # Fit the model and predict
        self.fit(x, y)
        return self.predict(Xnew)

    def multi_fit_predict(self, x=None, Y=None, Xnew=None):
        """
        Compute the conditional mean and return the smoothed function values
        at the points Xnew for each line of values in Y.

        This method is deprecated. Use FunctionEstimator.fit_reodict instead.

        Parameters
        ----------
        x : array-like, optional
            The training instances to estimate density function.
        Y : array-like, optional
            The training function values on cell states.
        Xnew : array-like, optional
            The new data to predict.

        Returns
        -------
        array-like
            The conditional mean function value at each test point in `x`.

        Raises
        ------
        ValueError
            If the number of samples in `x` and `Y` don't match, or if the
            number of features in `x` and `Xnew` don't match.
        """

        logger.warning(
            "Deprecation Warning: FunctionEstimator's multi_fit_predict method is deprecated. "
            "Use FunctionEstimator.fit_reodict instead."
        )

        # Set the x and validate inputs
        x = self.set_x(x)
        Y = _validate_array(Y, "Y")

        n_samples = x.shape[0]

        # Check for consistency in sample size between x and Y
        if Y.shape[0] != n_samples:
            if Y.shape[1] == n_samples:
                logger.warning(
                    "Y.shape[0] does not equal X.shape[0] (the number of samples). "
                    "However, Y.shape[1] == X.shape[0]. Transposing Y. "
                    "This assumes the columns of Y are the samples. Please verify."
                )
                Y = Y.T
        return self.fit_predict(x, Y, Xnew).T
