from .decomposition import DEFAULT_RANK, DEFAULT_METHOD
from .base_model import BaseEstimator, DEFAULT_COV_FUNC
from .inference import (
    compute_transform,
    compute_dimensionality_transform,
    compute_loss_func,
    compute_dimensionality_loss_func,
    compute_log_density_x,
    compute_conditional_mean,
    compute_conditional_mean_y,
    compute_conditional_mean_explog,
    DEFAULT_N_ITER,
    DEFAULT_INIT_LEARN_RATE,
    DEFAULT_JIT,
    DEFAULT_OPTIMIZER,
)
from .parameters import (
    compute_distances,
    compute_d,
    compute_d_factal,
    compute_mu,
    compute_initial_value,
    compute_initial_dimensionalities,
    compute_landmarks,
    DEFAULT_N_LANDMARKS,
)
from .util import (
    DEFAULT_JITTER,
    Log,
    local_dimensionality,
)
from .helper import vector_map
from .validation import (
    _validate_positive_int,
    _validate_positive_float,
    _validate_float,
    _validate_string,
    _validate_array,
)


DEFAULT_D_METHOD = "embedding"

logger = Log()


class DensityEstimator(BaseEstimator):
    R"""
    A class for non-parametric density estimation. It performs Bayesian inference with
    a Gaussian process prior and Nearest Neighbor likelihood. All intermediate computations
    are cached as instance variables, which allows viewing intermediate results and
    saving computation time by passing precomputed values as arguments to a new model.

    Parameters
    ----------
    cov_func_curry : function or type
        The generator of the Gaussian process covariance function.
        This must be a curry that takes one length scale argument and returns a
        covariance function of the form k(x, y) :math:`\rightarrow` float.
        Defaults to Matern52.
    n_landmarks : int
        The number of landmark points. If less than 1 or greater than or equal to the
        number of training points, inducing points will not be computed or used.
        Defaults to 5000.
    rank : int or float
        The rank of the approximate covariance matrix. If rank is an int, an :math:`n \times`
        rank matrix :math:`L` is computed such that :math:`L L^\top \approx K`, where `K` is the
        exact :math:`n \times n` covariance matrix. If rank is a float 0.0 :math:`\le` rank
        :math:`\le` 1.0, the rank/size of :math:`L` is selected such that the included eigenvalues
        of the covariance between landmark points account for the specified percentage of the sum
        of eigenvalues. Defaults to 0.99.
    method : str
        Determines how the rank is interpreted: as a fixed number of eigenvectors ('fixed'), a
        percent of eigenvalues ('percent'), or automatically ('auto'). If 'auto', the rank is
        interpreted as a fixed number of eigenvectors if it is an int and as a percent of
        eigenvalues if it is a float. This parameter is provided for clarity in the ambiguous case
        of 1 vs 1.0. Defaults to 'auto'.
    d_method : str
        The method to compute the intrinsic dimensionality of the data. Implemented options are
         - 'embedding': uses the embedding dimension `x.shape[1]`
         - 'fractal': uses the average fractal dimension (experimental)

        Defaults to 'embedding'.
    jitter : float
        A small amount added to the diagonal of the covariance matrix to bind eigenvalues
        numerically away from 0, ensuring numerical stability. Defaults to 1e-6.
    optimizer : str
        The optimizer for the maximum a posteriori or posterior density estimation. Options are
        'L-BFGS-B', stochastic optimizer 'adam', or automatic differentiation variational
        inference 'advi'. Defaults to 'L-BFGS-B'.
    n_iter : int
        The number of optimization iterations. Defaults to 100.
    init_learn_rate : float
        The initial learning rate. Defaults to 1.
    landmarks : array-like or None
        The points used to quantize the data for the approximate covariance. If None,
        landmarks are set as k-means centroids with k=n_landmarks. This is ignored if n_landmarks
        is greater than or equal to the number of training points. Defaults to None.
    nn_distances : array-like or None
        The nearest neighbor distances at each data point. If None, the nearest neighbor
        distances are computed automatically, using a KDTree if the dimensionality of the data
        is less than 20, or a BallTree otherwise. Defaults to None.
    d : int, array-like or None
        The intrinsic dimensionality of the data, i.e., the dimensionality of the embedded
        manifold. If None, `d` is set to the size of axis 1 of the training data points.
        Defaults to None.
    mu : float or None
        The mean :math:`\mu` of the Gaussian process. If None, sets :math:`\mu` to the 1st
        percentile of :math:`\text{mle}(\text{nn_distances}, d) - 10`, where :math:`\text{mle} =
        \log(\text{gamma}(d/2 + 1)) - (d/2) \cdot \log(\pi) - d \cdot \log(\text{nn_distances})`.
        Defaults to None.
    ls : float or None
        The length scale of the Gaussian process covariance function. If None, `ls` is set to
        the geometric mean of the nearest neighbor distances times a constant. If `cov_func`
        is supplied explicitly, `ls` has no effect. Defaults to None.
    cov_func : function or None
        The Gaussian process covariance function of the form k(x, y) :math:`\rightarrow` float.
        If None, the covariance function `cov_func` is automatically generated as `cov_func_curry(ls)`.
        Defaults to None.
    L : array-like or None
        A matrix such that :math:`L L^\top \approx K`, where :math:`K` is the covariance matrix.
        If None, `L` is computed automatically. Defaults to None.
    initial_value : array-like or None
        The initial guess for optimization. If None, the value :math:`z` that minimizes
        :math:`||Lz + \mu - mle|| + ||z||` is found, where :math:`\text{mle} = \log(\text{gamma}(d/2 + 1))
        - (d/2) \cdot \log(\pi) - d \cdot \log(\text{nn_distances})` and :math:`d` is the intrinsic
        dimensionality of the data. Defaults to None.
    jit : bool
        Use jax just-in-time compilation for loss and its gradient during optimization.
        Defaults to False.

    Attributes
    ----------
    cov_func_curry : function
        The generator of the Gaussian process covariance function.
    n_landmarks : int
        The number of landmark points.
    rank : int or float
        The rank of the approximate covariance matrix.
    method : str
        Determines how the rank is interpreted: as a fixed number of eigenvectors ('fixed'), a
        percent of eigenvalues ('percent'), or automatically ('auto').
    d_method : str
        The method to compute the intrinsic dimensionality of the data.
    jitter : float
        A small amount added to the diagonal of the covariance matrix.
    n_iter : int
        The number of optimization iterations.
    init_learn_rate : float
        The initial learning rate.
    landmarks : array-like
        The points used to quantize the data for the approximate covariance.
    nn_distances : array-like
        The nearest neighbor distances at each data point.
    d : int
        The intrinsic dimensionality of the data.
    mu : float
        The mean of the Gaussian process.
    ls : float
        The length scale of the Gaussian process covariance function.
    cov_func : function
        The Gaussian process covariance function of the form k(x, y) :math:`\rightarrow` float.
    L : array-like
        A matrix such that :math:`L L^\top \approx K`, where :math:`K` is the covariance matrix.
    initial_value : array-like
        The initial guess for optimization.
    jit : bool
        Use jax just-in-time compilation for loss and its gradient during optimization.
    x : ndarray
        Training data.
    transform : function
        Data transformation function applied before modeling.
    loss_func : function
        Loss function used for optimization.
    pre_transformation : ndarray
        Data after being preprocessed but before being transformed.
    opt_state : OptimizeResult
        The result of the optimization.
    losses : list of float
        The loss for each iteration during optimization.
    log_density_x : float
        Logarithmic density of training data.
    log_density_func: mellon.Predictor
        An instance of :class:`mellon.Predictor` that computes the log density
        at arbitrary prediction points. Provides methods for gradient and
        Hessian computations, and has serialization/deserialization features.
        Refer to :class:`mellon.Predictor` documentation for more details.
    """

    def __init__(
        self,
        cov_func_curry=DEFAULT_COV_FUNC,
        n_landmarks=DEFAULT_N_LANDMARKS,
        rank=DEFAULT_RANK,
        method=DEFAULT_METHOD,
        d_method=DEFAULT_D_METHOD,
        jitter=DEFAULT_JITTER,
        optimizer=DEFAULT_OPTIMIZER,
        n_iter=DEFAULT_N_ITER,
        init_learn_rate=DEFAULT_INIT_LEARN_RATE,
        landmarks=None,
        nn_distances=None,
        d=None,
        mu=None,
        ls=None,
        ls_factor=1,
        cov_func=None,
        L=None,
        initial_value=None,
        jit=DEFAULT_JIT,
    ):
        super().__init__(
            cov_func_curry=cov_func_curry,
            n_landmarks=n_landmarks,
            rank=rank,
            jitter=jitter,
            optimizer=optimizer,
            n_iter=n_iter,
            init_learn_rate=init_learn_rate,
            landmarks=landmarks,
            nn_distances=nn_distances,
            d=d,
            mu=mu,
            ls=ls,
            ls_factor=ls_factor,
            cov_func=cov_func,
            L=L,
            initial_value=initial_value,
            jit=jit,
        )
        self.d_method = _validate_string(
            d_method, "d_method", choices={"fractal", "embedding"}
        )
        self.transform = None
        self.loss_func = None
        self.opt_state = None
        self.losses = None
        self.pre_transformation = None
        self.log_density_x = None
        self.log_density_func = None

    def __repr__(self):
        name = self.__class__.__name__
        string = (
            f"{name}("
            f"cov_func_curry={self.cov_func_curry}, "
            f"n_landmarks={self.n_landmarks}, "
            f"rank={self.rank}, "
            f"method='{self.method}', "
            f"jitter={self.jitter}, "
            f"optimizer='{self.optimizer}', "
            f"n_iter={self.n_iter}, "
            f"init_learn_rate={self.init_learn_rate}, "
            f"landmarks={self.landmarks}, "
        )
        if self.nn_distances is None:
            string += "nn_distances=None, "
        else:
            string += "nn_distances=nn_distances, "
        string += (
            f"d={self.d}, "
            f"mu={self.mu}, "
            f"ls={self.mu}, "
            f"cov_func={self.cov_func}, "
        )
        if self.L is None:
            string += "L=None, "
        else:
            string += "L=L, "
        if self.initial_value is None:
            string += "initial_value=None, "
        else:
            string += "initial_value=initial_value, "
        string += f"jit={self.jit}" ")"
        return string

    def _compute_d(self):
        x = self.x
        if self.d_method == "fractal":
            logger.warning("Using EXPERIMENTAL fractal dimensionality selection.")
            d = compute_d_factal(x)
            logger.info(f"Using d={d}.")
        else:
            d = compute_d(x)
        if d > 50:
            message = f"""The detected dimensionality of the data is over 50,
            which is likely to cause numerical instability issues.
            Consider running a dimensionality reduction algorithm, or
            if this number of dimensions is intended, explicitly pass
            d={self.d} as a parameter."""
            raise ValueError(message)
        return d

    def _compute_mu(self):
        nn_distances = self.nn_distances
        d = self.d
        mu = compute_mu(nn_distances, d)
        return mu

    def _compute_initial_value(self):
        nn_distances = self.nn_distances
        d = self.d
        mu = self.mu
        L = self.L
        initial_value = compute_initial_value(nn_distances, d, mu, L)
        return initial_value

    def _compute_transform(self):
        mu = self.mu
        L = self.L
        transform = compute_transform(mu, L)
        return transform

    def _compute_loss_func(self):
        nn_distances = self.nn_distances
        d = self.d
        transform = self.transform
        k = self.initial_value.shape[0]
        loss_func = compute_loss_func(nn_distances, d, transform, k)
        return loss_func

    def _set_log_density_x(self):
        pre_transformation = self.pre_transformation
        transform = self.transform
        log_density_x = compute_log_density_x(pre_transformation, transform)
        self.log_density_x = log_density_x

    def _set_log_density_func(self):
        x = self.x
        landmarks = self.landmarks
        pre_transformation = self.pre_transformation
        log_density_x = self.log_density_x
        mu = self.mu
        cov_func = self.cov_func
        jitter = self.jitter
        logger.info("Computing predictive function.")
        log_density_func = compute_conditional_mean(
            x,
            landmarks,
            pre_transformation,
            log_density_x,
            mu,
            cov_func,
            jitter=jitter,
        )
        self.log_density_func = log_density_func

    def prepare_inference(self, x):
        R"""
        Set all attributes in preparation for optimization, but do not
        perform Bayesian inference. It is not necessary to call this
        function before calling fit.

        :param x: The training instances to estimate density function.
        :type x: array-like
        :return: loss_func, initial_value - The Bayesian loss function and
            initial guess for optimization.
        :rtype: function, array-like
        """
        self._set_x(x)
        self._prepare_attribute("nn_distances")
        self._prepare_attribute("d")
        self._prepare_attribute("mu")
        self._prepare_attribute("ls")
        self._prepare_attribute("cov_func")
        self._prepare_attribute("landmarks")
        self._prepare_attribute("L")
        self._prepare_attribute("initial_value")
        self._prepare_attribute("transform")
        self._prepare_attribute("loss_func")
        return self.loss_func, self.initial_value

    def run_inference(self, loss_func=None, initial_value=None, optimizer=None):
        R"""
        Perform Bayesian inference, optimizing the pre_transformation parameters.
        If you would like to run your own inference procedure, use the loss_function
        and initial_value attributes and set pre_transformation to the optimized
        parameters.

        :param loss_func: The Bayesian loss function. If None, uses the stored
            loss_func attribute.
        :type loss_func: function
        :param initial_value: The initial guess for optimization. If None, uses
            the stored initial_value attribute.
        :type initial_value: array-like
        :return: pre_transformation - The optimized parameters.
        :rtype: array-like
        """
        if loss_func is not None:
            self.loss_func = loss_func
        if initial_value is not None:
            self.initial_value = initial_value
        if optimizer is not None:
            self.optimizer = optimizer
        self._run_inference()
        return self.pre_transformation

    def process_inference(self, pre_transformation=None, build_predict=True):
        R"""
        Use the optimized parameters to compute the log density at the
        training points. If build_predict, also build the prediction function.

        :param pre_transformation: The optimized parameters. If None, uses the stored
            pre_transformation attribute.
        :type pre_transformation: array-like
        :param build_predict: Whether or not to build the prediction function.
            Defaults to True.
        :type build_predict: bool
        :return: log_density_x - The log density
        :rtype: array-like
        """
        if pre_transformation is not None:
            self.pre_transformation = _validate_array(
                pre_transformation, "pre_transformation"
            )
        self._set_log_density_x()
        if build_predict:
            self._set_log_density_func()
        return self.log_density_x

    def fit(self, x=None, build_predict=True):
        R"""
        Fit the model from end to end.

        :param x: The training instances to estimate density function.
        :type x: array-like
        :param build_predict: Whether or not to build the prediction function.
            Defaults to True.
        :type build_predict: bool
        :return: self - A fitted instance of this estimator.
        :rtype: Object
        """
        if self.x is not None and self.x is not x:
            message = "self.x has been set already, but is not equal to the argument x."
            raise ValueError(message)
        if self.x is None and x is None:
            message = "Required argument x is missing and self.x has not been set."
            raise ValueError(message)
        if x is None:
            x = self.x
        else:
            x = _validate_array(x, "x")

        self.prepare_inference(x)
        self.run_inference()
        self.process_inference(build_predict=build_predict)
        return self

    @property
    def predict(self):
        R"""
    An instance of the :class:`mellon.Predictor` that predicts the log density at each point in x.

        It contains a __call__ method which can be used to predict the log density.
        The instance also supports serialization features which allows for saving
        and loading the predictor state. Refer to mellon.Predictor documentation for more details.

        :param x: The new data to predict.
        :type x: array-like
        :return: log_density - The log density at each test point in x.
        :rtype: array-like
        """
        if self.log_density_func is None:
            self._set_log_density_func()
        return self.log_density_func

    def fit_predict(self, x=None, build_predict=False):
        R"""
        Perform Bayesian inference and return the log density at training points.

        :param x: The training instances to estimate density function.
        :type x: array-like
        :return: log_density_x - The log density at each training point in x.
        """
        if self.x is not None and self.x is not x:
            message = "self.x has been set already, but is not equal to the argument x."
            error = ValueError(message)
            logger.error(error)
            raise error
        if self.x is None and x is None:
            message = "Required argument x is missing and self.x has not been set."
            error = ValueError(message)
            logger.error(error)
            raise error
        if x is None:
            x = self.x
        else:
            x = _validate_array(x, "x")

        self.fit(x, build_predict=build_predict)
        return self.log_density_x


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
        The length scale of the Gaussian process covariance function. If None, sets ls to the
        geometric mean of the nearest neighbor distances times a constant. This has no effect
        if cov_func is supplied explicitly. Defaults to None.
    cov_func : function or None, optional
        The Gaussian process covariance function of the form k(x, y) :math:`\rightarrow` float.
        If None, automatically generates the covariance function cov_func = cov_func_curry(ls).
        Defaults to None.
    sigma : float, optional
        The standard deviation of the white noise. Defaults to 0.
    jit : bool, optional
        Use JAX just-in-time compilation for the loss function and its gradient during optimization.
        Defaults to False.

    Attributes
    ----------
    n_landmarks : int
        The number of landmark points.
    jitter : float
        A small amount added to the diagonal of the covariance matrix for numerical stability.
    landmarks : array-like
        The points used to quantize the data.
    nn_distances : array-like
        The nearest neighbor distances for each data point.
    mu : float
        The mean of the Gaussian process :math:`\mu`.
    ls : float
        The length scale of the Gaussian process covariance function.
    ls_factor : float
        Factor used to scale the automatically selected length scale. Defaults to 1.
    cov_func : function
        The Gaussian process covariance function.
    sigma : float
        Standard deviation of the white noise.
    x : array-like
        The cell states.
    y : array-like
        Function values on the cell states.
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

        return vector_map(conditional, Y, do_jit=jit)


class DimensionalityEstimator(BaseEstimator):
    R"""
    This class provides a non-parametric method for estimating local dimensionality and density.
    It uses Bayesian inference with a Gaussian process prior and a normal distribution for local scaling rates.
    The class caches all intermediate computations as instance variables, enabling users to view intermediate results and
    save computational time by passing precomputed values to a new model instance.

    Parameters
    ----------
    cov_func_curry: function or type, optional (default=Matern52)
        A generator for the Gaussian process covariance function. It should be a curry function taking one length scale argument
        and returning a covariance function of the form k(x, y) :math:`\rightarrow` float.

    n_landmarks: int, optional (default=5000)
        The number of landmark points. If less than 1 or greater than or equal to the number of training points,
        inducing points are not computed or used.

    rank: int or float, optional (default=0.99)
        The rank of the approximate covariance matrix. When interpreted as an integer, an :math:`n \times` rank matrix
        :math:`L` is computed such that :math:`L L^\top \approx K`, where :math:`K` is the exact :math:`n \times n` covariance matrix.
        When interpreted as a float (between 0.0 and 1.0), the rank/size of :math:`L` is chosen such that the included eigenvalues of the covariance
        between landmark points account for the specified percentage of the total eigenvalues.

    method: str, optional (default='auto')
        Determines whether the `rank` parameter is interpreted as a fixed number of eigenvectors ('fixed'), a percentage of eigenvalues ('percent'),
        or determined automatically ('auto'). In 'auto' mode, `rank` is treated as a fixed number if it is an integer, or a percentage if it's a float.

    jitter: float, optional (default=1e-6)
        A small amount added to the diagonal of the covariance matrix to ensure numerical stability by keeping eigenvalues away from 0.

    optimizer: str, optional (default='L-BFGS-B')
        The optimizer to use for maximum a posteriori density estimation. It can be either 'L-BFGS-B' or 'adam'.

    n_iter: int, optional (default=100)
        The number of iterations for optimization.

    init_learn_rate: float, optional (default=1)
        The initial learning rate for the optimizer.

    landmarks: array-like or None, optional
        Points used to quantize the data for approximate covariance. If None, landmarks are set as k-means centroids
        with k=n_landmarks. If the number of landmarks is greater than or equal to the number of training points, this parameter is ignored.

    k: int, optional (default=10)
        The number of nearest neighbor distances to consider.

    distances: array-like or None, optional
        The k nearest neighbor distances at each data point. If None, these distances are computed automatically using KDTree (if data dimensionality is < 20)
        or BallTree otherwise.

    d: array-like, optional
        The estimated local intrinsic dimensionality of the data. This is only used to initialize the density estimation.
        If None, an empirical estimate is used.

    mu_dim: float or None, optional (default=0)
        The mean of the Gaussian process for log intrinsic dimensionality :math:`\mu_D`.

    mu_dens: float or None, optional
        The mean of the Gaussian process for log density :math:`\mu_\rho`. If None,
        `mu_dens` is set to the 1st percentile of :math:`\text{mle}(\text{nn_distances}, d) - 10`.

    ls: float or None, optional
        The length scale of the Gaussian process covariance function. If None, it's set to the geometric mean of the nearest neighbor distances multiplied by a constant.
        If `cov_func` is supplied explicitly, `ls` has no effect.

    cov_func: function or None, optional
        The Gaussian process covariance function of the form k(x, y) :math:`\rightarrow` float. If None, the covariance function is generated automatically as `cov_func = cov_func_curry(ls)`.

    L: array-like or None, optional
        A matrix such that :math:`L L^\top \approx K`, where :math:`K` is the covariance matrix. If None, `L` is computed automatically.

    initial_value: array-like or None, optional
        The initial guess for optimization. If None, an optimized :math:`z` is found that minimizes :math:`||Lz + \mu_\cdot - mle|| + ||z||`, where :math:`\text{mle}` is the maximum likelihood estimate
        for density initialization and the neighborhood-based local intrinsic dimensionality for dimensionality initialization.

    jit: bool, optional (default=False)
        If True, use JAX's just-in-time compilation for loss and its gradient during optimization.

    Attributes
    ----------
    The attributes of this class correspond to the parameters of the same names, with the following additional attributes:

    x: array-like
        The training data.

    transform: function
        Used to map the latent representation to the log-density on the training data.

    loss_func: function
        The Bayesian loss function.

    pre_transformation: array-like
        The optimized parameters before transformation, used to map the latent representation to the log-density on the training data.

    opt_state: object
        The final state of the optimizer.

    losses: list or float
        The history of losses throughout training with 'adam' or final loss with 'L-BFGS-B'.

    local_dim_x: array-like
        The local intrinsic dimensionality at the training points.

    log_density_x: array-like
        The log density with varying units at the training points. Density indicates the number of cells per volume in state space.
        As the intrinsic dimensionality of the volume changes, the resulting density unit varies.

    local_dim_func: mellon.Predictor
        An instance of :class:`mellon.Predictor` that computes the local intrinsic dimensionality
        at arbitrary prediction points. Provides methods for gradient and
        Hessian computations, and has serialization/deserialization features.
        Refer to :class:`mellon.Predictor` documentation for more details.


    log_density_func: mellon.Predictor
        An instance of :class:`mellon.Predictor` that computes the log density with varying units
        at arbitrary prediction points. Provides methods for gradient and
        Hessian computations, and has serialization/deserialization features.
        Refer to :class:`mellon.Predictor` documentation for more details.
    """

    def __init__(
        self,
        cov_func_curry=DEFAULT_COV_FUNC,
        n_landmarks=DEFAULT_N_LANDMARKS,
        rank=DEFAULT_RANK,
        method=DEFAULT_METHOD,
        jitter=DEFAULT_JITTER,
        optimizer=DEFAULT_OPTIMIZER,
        n_iter=DEFAULT_N_ITER,
        init_learn_rate=DEFAULT_INIT_LEARN_RATE,
        landmarks=None,
        k=10,
        distances=None,
        d=None,
        mu_dim=0,
        mu_dens=None,
        ls=None,
        ls_factor=1,
        cov_func=None,
        L=None,
        initial_value=None,
        jit=DEFAULT_JIT,
    ):
        super().__init__(
            cov_func_curry=cov_func_curry,
            n_landmarks=n_landmarks,
            rank=rank,
            method=method,
            jitter=jitter,
            optimizer=optimizer,
            n_iter=n_iter,
            init_learn_rate=init_learn_rate,
            landmarks=landmarks,
            nn_distances=None,
            d=d,
            mu=mu_dens,
            ls=ls,
            ls_factor=ls_factor,
            cov_func=cov_func,
            L=L,
            initial_value=initial_value,
            jit=jit,
        )
        self.k = _validate_positive_int(k, "k")
        self.mu_dim = _validate_float(mu_dim, "mu_dim")
        self.mu_dens = _validate_float(mu_dens, "mu_dens", optional=True)
        self.distances = _validate_array(distances, "distances", optional=True)
        self.transform = None
        self.loss_func = None
        self.opt_state = None
        self.losses = None
        self.pre_transformation = None
        self.local_dim_x = None
        self.log_density_x = None
        self.local_dim_func = None
        self.log_density_func = None

    def __repr__(self):
        name = self.__class__.__name__
        string = (
            f"{name}("
            f"cov_func_curry={self.cov_func_curry}, "
            f"n_landmarks={self.n_landmarks}, "
            f"rank={self.rank}, "
            f"method='{self.method}', "
            f"jitter={self.jitter}, "
            f"optimizer='{self.optimizer}', "
            f"n_iter={self.n_iter}, "
            f"init_learn_rate={self.init_learn_rate}, "
            f"landmarks={self.landmarks}, "
        )
        if self.distances is None:
            string += "distances=None, "
        else:
            string += "distances=distances, "
        string += (
            f"d={self.d}, "
            f"mu_dim={self.mu_dim}, "
            f"mu_dens={self.mu_dens}, "
            f"ls={self.mu}, "
            f"cov_func={self.cov_func}, "
        )
        if self.L is None:
            string += "L=None, "
        else:
            string += "L=L, "
        if self.initial_value is None:
            string += "initial_value=None, "
        else:
            string += "initial_value=initial_value, "
        string += f"jit={self.jit}" ")"
        return string

    def _compute_mu_dens(self):
        nn_distances = self.nn_distances
        d = self.d
        mu = compute_mu(nn_distances, d)
        return mu

    def _compute_d(self):
        x = self.x
        d = local_dimensionality(x, neighbor_idx=None)
        return d

    def _compute_initial_value(self):
        x = self.x
        d = self.d
        nn_distances = self.nn_distances
        mu_dim = self.mu_dim
        mu_dens = self.mu_dens
        L = self.L
        initial_value = compute_initial_dimensionalities(
            x, mu_dim, mu_dens, L, nn_distances, d
        )
        return initial_value

    def _compute_transform(self):
        mu_dim = self.mu_dim
        mu_dens = self.mu_dens
        L = self.L
        transform = compute_dimensionality_transform(mu_dim, mu_dens, L)
        return transform

    def _compute_distances(self):
        x = self.x
        k = self.k
        logger.info("Computing distances.")
        distances = compute_distances(x, k=k)
        return distances

    def _compute_nn_distances(self):
        distances = self.distances
        return distances[:, 0]

    def _compute_loss_func(self):
        distances = self.distances
        transform = self.transform
        k = self.initial_value.shape[0]
        loss_func = compute_dimensionality_loss_func(distances, transform, k)
        return loss_func

    def _set_local_dim_x(self):
        pre_transformation = self.pre_transformation
        transform = self.transform
        local_dim_x, log_density_x = compute_log_density_x(
            pre_transformation, transform
        )
        self.local_dim_x = local_dim_x
        self.log_density_x = log_density_x

    def _set_local_dim_func(self):
        x = self.x
        landmarks = self.landmarks
        local_dim_x = self.local_dim_x
        mu = self.mu_dim
        cov_func = self.cov_func
        jitter = self.jitter
        logger.info("Computing predictive dimensionality function.")
        log_dim_func = compute_conditional_mean_explog(
            x,
            landmarks,
            local_dim_x,
            mu,
            cov_func,
            jitter=jitter,
        )
        self.local_dim_func = log_dim_func

    def _set_log_density_func(self):
        x = self.x
        landmarks = self.landmarks
        pre_transformation = self.pre_transformation
        log_density_x = self.log_density_x
        mu = self.mu_dens
        cov_func = self.cov_func
        jitter = self.jitter
        logger.info("Computing predictive density function.")
        log_density_func = compute_conditional_mean(
            x,
            landmarks,
            pre_transformation,
            log_density_x,
            mu,
            cov_func,
            jitter=jitter,
        )
        self.log_density_func = log_density_func

    def prepare_inference(self, x):
        R"""
        Set all attributes in preparation for optimization, but do not
        perform Bayesian inference. It is not necessary to call this
        function before calling fit.

        :param x: The training instances to estimate density function.
        :type x: array-like
        :return: loss_func, initial_value - The Bayesian loss function and
            initial guess for optimization.
        :rtype: function, array-like
        """
        self._set_x(x)
        self._prepare_attribute("distances")
        self._prepare_attribute("nn_distances")
        self._prepare_attribute("d")
        self._prepare_attribute("mu_dens")
        self._prepare_attribute("ls")
        self._prepare_attribute("cov_func")
        self._prepare_attribute("landmarks")
        self._prepare_attribute("L")
        self._prepare_attribute("initial_value")
        self._prepare_attribute("transform")
        self._prepare_attribute("loss_func")
        return self.loss_func, self.initial_value

    def run_inference(self, loss_func=None, initial_value=None, optimizer=None):
        R"""
        Perform Bayesian inference, optimizing the pre_transformation parameters.
        If you would like to run your own inference procedure, use the loss_function
        and initial_value attributes and set pre_transformation to the optimized
        parameters.

        :param loss_func: The Bayesian loss function. If None, uses the stored
            loss_func attribute.
        :type loss_func: function
        :param initial_value: The initial guess for optimization. If None, uses
            the stored initial_value attribute.
        :type initial_value: array-like
        :return: pre_transformation - The optimized parameters.
        :rtype: array-like
        """
        if loss_func is not None:
            self.loss_func = loss_func
        if initial_value is not None:
            self.initial_value = initial_value
        if optimizer is not None:
            self.optimizer = optimizer
        self._run_inference()
        return self.pre_transformation

    def process_inference(self, pre_transformation=None, build_predict=True):
        R"""
        Use the optimized parameters to compute the local dimensionality at the
        training points. If build_predict, also build the prediction function.

        :param pre_transformation: The optimized parameters. If None, uses the stored
            pre_transformation attribute.
        :type pre_transformation: array-like
        :param build_predict: Whether or not to build the prediction function.
            Defaults to True.
        :type build_predict: bool
        :return: local_dim_x - The local dimensionality
        :rtype: array-like
        """
        if pre_transformation is not None:
            self.pre_transformation = pre_transformation
        self._set_local_dim_x()
        if build_predict:
            self._set_local_dim_func()
            self._set_log_density_func()
        return self.local_dim_x, self.log_density_x

    def fit(self, x=None, build_predict=True):
        R"""
        Fit the model from end to end.

        :param x: The training instances to estimate dimensionality function.
        :type x: array-like
        :param build_predict: Whether or not to build the prediction function.
            Defaults to True.
        :type build_predict: bool
        :return: self - A fitted instance of this estimator.
        :rtype: Object
        """
        if self.x is not None and self.x is not x:
            message = "self.x has been set already, but is not equal to the argument x."
            raise ValueError(message)
        if self.x is None and x is None:
            message = "Required argument x is missing and self.x has not been set."
            raise ValueError(message)
        if x is None:
            x = self.x
        else:
            x = _validate_array(x, "x")

        self.prepare_inference(x)
        self.run_inference()
        self.process_inference(build_predict=build_predict)
        return self

    @property
    def predict_density(self):
        R"""
        Predict the log density with adaptive unit at each point in x.
        Note that the unit of denity depends on the dimensionality of the
        volume.

        :param x: The new data to predict.
        :type x: array-like
        :return: log_density - The log density at each test point in x.
        :rtype: array-like
        """
        if self.log_density_func is None:
            self._set_log_density_func()
        return self.log_density_func

    @property
    def predict(self):
        R"""
        An instance of the :class:`mellon.Predictor` that predicts the dimensionality at each point in x.

        It contains a __call__ method which can be used to predict the dimensionality.
        The instance also supports serialization features which allows for saving
        and loading the predictor state. Refer to mellon.Predictor documentation for more details.

        :param x: The new data to predict.
        :type x: array-like
        :return: dimensionality - The dimensionality at each test point in x.
        :rtype: array-like
        """
        if self.local_dim_func is None:
            self._set_local_dim_func()
        return self.local_dim_func

    def fit_predict(self, x=None, build_predict=False):
        R"""
        Perform Bayesian inference and return the local dimensionality at training points.

        :param x: The training instances to estimate the local dimensionality function.
        :type x: array-like
        :return: local_dim_x - The local dimensionality at each training point in x.
        """
        if self.x is not None and self.x is not x:
            message = "self.x has been set already, but is not equal to the argument x."
            error = ValueError(message)
            logger.error(error)
            raise error
        if self.x is None and x is None:
            message = "Required argument x is missing and self.x has not been set."
            error = ValueError(message)
            logger.error(error)
            raise error
        if x is None:
            x = self.x
        else:
            x = _validate_array(x, "x")

        self.fit(x, build_predict=build_predict)
        return self.local_dim_x
