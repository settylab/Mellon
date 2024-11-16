import logging
from .base_model import BaseEstimator, DEFAULT_COV_FUNC
from .inference import (
    compute_conditional,
    DEFAULT_N_ITER,
    DEFAULT_INIT_LEARN_RATE,
    DEFAULT_OPTIMIZER,
)
from .util import (
    DEFAULT_JITTER,
    GaussianProcessType,
    object_str,
    object_html,
)
from .validation import (
    validate_float_or_iterable_numerical,
    validate_float,
    validate_array,
    validate_bool,
)


DEFAULT_D_METHOD = "embedding"

logger = logging.getLogger("mellon")


class FunctionEstimator(BaseEstimator):
    R"""
    This class implements a Function Estimator that uses a conditional normal distribution
    to smoothen and extend a function on all cell states using the Mellon abstractions.

    Parameters
    ----------
    cov_func_curry : function or type
        A curry that takes one length scale argument and returns a covariance function
        of the form k(x, y) :math:`\rightarrow` float. Defaults to Matern52.

    n_landmarks : int
        The number of landmark/inducing points. Only used if a sparse GP is indicated
        through gp_type. If 0 or equal to the number of training points, inducing points
        will not be computed or used. Defaults to 5000.

    gp_type : str or GaussianProcessType
        The type of sparcification used for the Gaussian Process:
         - 'full' None-sparse Gaussian Process
         - 'sparse_cholesky' Sparse GP using landmarks/inducing points,
            typically employed to enable scalable GP models.

        The value can be either a string matching one of the above options or an instance of
        the `mellon.util.GaussianProcessType` Enum. If a partial match is found with the
        Enum, a warning will be logged, and the closest match will be used.
        Defaults to 'sparse_cholesky'.

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

    cov_func : mellon.Covaraince or None
        The Gaussian process covariance function as instance of :class:`mellon.Covaraince`.
        If None, the covariance function `cov_func` is automatically generated as `cov_func_curry(ls)`.
        Defaults to None.

    sigma : float, optional
        The standard deviation of the white noise. Defaults to 0.

    y_is_mean : bool
        Wether to consider y the GP mean or a noise measurment
        subject to `sigma` or `y_cov_factor`. Has no effect if `L` is passed.
        Defaults to False.

    predictor_with_uncertainty : bool
        If set to True, computes the predictor instance `.predict` with its predictive uncertainty.
        The uncertainty comes from two sources:

        1) `.predict.mean_covariance`:
            Uncertainty arising from the input noise `sigma`.

        2) `.predict.covariance`:
            Uncertainty for out-of-bag states originating from the compressed function representation
            in the Gaussian Process. Specifically, this uncertainty corresponds to locations that are
            not inducing points of the Gaussian Process and represents the covariance of the
            conditional normal distribution.

    jit : bool, optional
        Use JAX just-in-time compilation for the loss function and its gradient during optimization.
        Defaults to False.
    """

    def __init__(
        self,
        cov_func_curry=DEFAULT_COV_FUNC,
        n_landmarks=None,
        gp_type=None,
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
        y_is_mean=False,
        predictor_with_uncertainty=False,
        jit=True,
    ):
        super().__init__(
            cov_func_curry=cov_func_curry,
            n_landmarks=n_landmarks,
            rank=1.0,
            jitter=jitter,
            gp_type=gp_type,
            landmarks=landmarks,
            nn_distances=nn_distances,
            mu=mu,
            ls=ls,
            ls_factor=ls_factor,
            cov_func=cov_func,
            predictor_with_uncertainty=predictor_with_uncertainty,
            jit=jit,
        )
        self.y_is_mean = validate_bool(y_is_mean, "y_is_mean")
        self.mu = validate_float(mu, "mu")
        self.sigma = validate_float_or_iterable_numerical(sigma, "sigma", positive=True)
        if (
            self.gp_type == GaussianProcessType.FULL_NYSTROEM
            or self.gp_type == GaussianProcessType.SPARSE_NYSTROEM
        ):
            message = (
                f"gp_type={gp_type} but the Nyström rank reduction is "
                "not available for the Function Estimator. "
                "Use gp_type='cholesky' or gp_type='full' instead."
            )
            logger.error(message)
            raise ValueError(message)

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

    def __repr__(self):
        name = self.__class__.__name__
        landmarks = object_str(self.landmarks, ["landmarks", "dims"])
        Lp = object_str(self.Lp, ["landmarks", "landmarks"])
        L = object_str(self.L, ["cells", "ranks"])
        nn_distances = object_str(self.nn_distances, ["cells"])
        initial_value = object_str(self.initial_value, ["ranks"])
        d = object_str(self.d, ["cells"])
        string = (
            f"{name}("
            f"\n    check_rank={self.check_rank},"
            f"\n    cov_func={self.cov_func},"
            f"\n    cov_func_curry={self.cov_func_curry},"
            f"\n    d={d},"
            f"\n    gp_type={self.gp_type},"
            f"\n    initial_value={initial_value},"
            f"\n    jit={self.jit},"
            f"\n    jitter={self.jitter},"
            f"\n    landmarks={landmarks},"
            f"\n    L={L},"
            f"\n    Lp={Lp},"
            f"\n    ls={self.ls},"
            f"\n    ls_factor={self.ls_factor},"
            f"\n    mu={self.mu},"
            f"\n    n_landmarks={self.n_landmarks},"
            f"\n    nn_distances={nn_distances},"
            f"\n    optimizer={self.optimizer},"
            f"\n    predictor_with_uncertainty={self.predictor_with_uncertainty},"
            f"\n    rank={self.rank},"
            f"\n    sigma={self.sigma},"
            f"\n    y_is_mean={self.y_is_mean},"
            "\n)"
        )
        return string

    def _repr_html_(self):
        """
        Generate an HTML representation for the FunctionEstimator subclass for display in Jupyter.
        """

        # Header with class name and description
        header = f"""
        <h2>Function Estimator: {self.__class__.__name__}</h2>
        <p><em>Function on all cell states using Gaussian Process and Mellon abstractions.</em></p>
        """

        # Core attributes as a list
        core_attributes = f"""
        <h3>Core Attributes</h3>
        <ul>
            <li><strong>Covariance Function:</strong> {object_html(self.cov_func or 'Not Set')}</li>
            <li><strong>Optimizer:</strong> {self.optimizer}</li>
            <li><strong>Number of Landmarks:</strong> {self.n_landmarks or 'Not Set'}</li>
            <li><strong>Gaussian Process Type:</strong> {self.gp_type or 'Not Set'}</li>
            <li><strong>Predictor with Uncertainty:</strong> {'Yes' if self.predictor_with_uncertainty else 'No'}</li>
        </ul>
        """

        # Table of model parameters with compact columns
        parameters = {
            "Jitter": self.jitter,
            "Mean (μ)": self.mu or "Not Set",
            "Length Scale (ls)": self.ls or "Not Set",
            "Length-Scale Factor": self.ls_factor,
            "Noise Standard Deviation (σ)": self.sigma,
            "y_is_mean": self.y_is_mean,
            "Nearest Neighbor Distances": self.nn_distances,
        }
        parameters_table = f"""
        <h3>Model Parameters</h3>
        <table style="border: 1px solid black; border-collapse: collapse; width: auto;">
            <tr><th style="border: 1px solid black; text-align: left;">Parameter</th><th style="border: 1px solid black; text-align: left;">Value</th></tr>
            {''.join(f'<tr><td style="border: 1px solid black; text-align: left;">{key}</td><td style="border: 1px solid black; text-align: left;">{object_html(value)}</td></tr>' for key, value in parameters.items())}
        </table>
        """

        # Predictor status
        predictor_status = (
            "<p style='color:green;'><strong>Predictor:</strong> Available</p>"
            if hasattr(self, "conditional") and self.conditional
            else "<p style='color:red;'><strong>Predictor:</strong> Not Yet Computed</p>"
        )

        # Details about the predictor if available
        predictor_details = ""
        if hasattr(self, "conditional") and self.conditional:
            predictor_details = f"""
            <h3>Predictor Details</h3>
            <p><strong>Predictor Instance:</strong> {object_html(self.conditional)}</p>
            """

        # Combine all sections
        return (
            header
            + core_attributes
            + parameters_table
            + predictor_status
            + predictor_details
        )

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
        self._prepare_attribute("n_landmarks")
        self._prepare_attribute("gp_type")
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
            x = validate_array(x, "x")
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
        y_is_mean = self.y_is_mean
        with_uncertainty = self.predictor_with_uncertainty
        conditional = compute_conditional(
            x,
            landmarks,
            None,
            None,
            y,
            mu,
            cov_func,
            None,
            None,
            sigma,
            jitter=jitter,
            y_is_mean=y_is_mean,
            with_uncertainty=with_uncertainty,
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
        y = validate_array(y, "y")

        n_samples = x.shape[0]
        # Check if the number of samples in x and y match
        if y.shape[0] != n_samples:
            raise ValueError(
                f"X.shape[0] = {n_samples:,} (n_samples) should equal "
                f"y.shape[0] = {y.shape[0]:,}."
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
        y = validate_array(y, "y")
        Xnew = validate_array(Xnew, "Xnew", optional=True)

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
        Y = validate_array(Y, "Y")

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
