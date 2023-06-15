from .decomposition import DEFAULT_RANK, DEFAULT_METHOD
from .base_model import BaseEstimator, DEFAULT_COV_FUNC
from .inference import (
    compute_transform,
    compute_loss_func,
    compute_log_density_x,
    compute_conditional_mean_times,
    DEFAULT_N_ITER,
    DEFAULT_INIT_LEARN_RATE,
    DEFAULT_JIT,
    DEFAULT_OPTIMIZER,
)
from .parameters import (
    compute_nn_distances_within_time_points,
    compute_cov_func,
    compute_d,
    compute_d_factal,
    compute_mu,
    compute_initial_value,
    DEFAULT_N_LANDMARKS,
)
from .compute_ls_time import compute_ls_time
from .util import (
    DEFAULT_JITTER,
    Log,
)
from .validation import (
    _validate_time_x,
    _validate_positive_float,
    _validate_string,
    _validate_array,
)


DEFAULT_D_METHOD = "embedding"

logger = Log()


class TimeSensitiveDensityEstimator(BaseEstimator):
    R"""
    A class for non-parametric density estimation with time sensitivity.
    It performs Bayesian inference with
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
        The nearest neighbor distances at each data point within each time point.
        If None, the nearest neighbor
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

    ls : float or None, optional
        The length scale for the Gaussian process covariance function.
        If None (default), the length scale is automatically selected based on
        a heuristic link between the nearest neighbor distances and the optimal
        length scale.

    ls_time : float or None
        The length scale of the Gaussian process covariance function for the time dimension.
        If None, `ls_time` is set to the length scale that best induces a covariance
        (using the `cov_func_curry`) between the time points that best mimics the
        Pearson correlation observed between densities of the individual time points. If `cov_func`
        is supplied explicitly, `ls_time` has no effect. Defaults to None.

    ls_factor : float, optional
        A scaling factor applied to the length scale when it's automatically
        selected. It is used to manually adjust the automatically chosen length

    ls_time_factor : float, optional
        A scaling factor applied to the time length scale (`ls_time`) when it's automatically
        selected. This allows for manual adjustment of the automatically determined time length scale.
        Defaults to 1.

    density_estimator_kwargs : dict, optional
        A dictionary of keyword arguments to be passed for timepoint-specific
        density estimation during the automatic selection of the time length scale `ls_time`.
        This parameter allows custom configuration for the density estimation process.
        Note that this parameter has no effect if `ls_time` is specified explicitly.
        Default is an empty dictionary ({}).

    cov_func : mellon.Covariance or None
        The Gaussian process covariance function of the form k(x, y) :math:`\rightarrow` float.
        Should be an instance of a class that inherits from :class:`mellon.Covariance`.
        If None, the covariance function `cov_func` is automatically generated as
        `cov_func_curry(ls, active_dims=slice(None, -1)) * cov_func_curry(ls_time, active_dims=-1)`.
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
        The nearest neighbor distances at each data point within their respective time point.
    d : int
        The intrinsic dimensionality of the data.
    mu : float
        The mean of the Gaussian process.
    ls : float
        The length scale of the Gaussian process covariance function for spacial dimensions.
    ls_time : float
        The length scale of the Gaussian process covariance function for the time dimension.
    ls_factor : float
        Factor used to scale the automatically selected length scale.
    ls_time_factor : float
        Factor used to scale the automatically selected time length scale.
    density_estimator_kwargs : dict
        A dictionary of keyword arguments for the density estimation within time points.
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
        ls_time=None,
        ls_factor=1,
        ls_factor_times=1,
        density_estimator_kwargs=dict(),
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
        if not isinstance(density_estimator_kwargs, dict):
            raise ValueError("density_estimator_kwargs needs to be a dictionary.")
        self.density_estimator_kwargs = density_estimator_kwargs
        self.d_method = _validate_string(
            d_method, "d_method", choices={"fractal", "embedding"}
        )
        self.ls_time = _validate_positive_float(ls_time, "ls_time", optional=True)
        self.ls_factor_times = _validate_positive_float(
            ls_factor_times, "ls_factor_times"
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
            f"ls={self.ls}, "
            f"ls_time={self.ls_time}, "
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
        x = self.x[:, :-1]
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

    def _compute_nn_distances(self):
        x = self.x
        logger.info("Computing nearest neighbor distances within time points.")
        nn_distances = compute_nn_distances_within_time_points(x)
        return nn_distances

    def _compute_ls_time(self):
        nn_distances = self.nn_distances
        x = self.x
        cov_func_curry = self.cov_func_curry
        density_estimator_kwargs = self.density_estimator_kwargs
        logger.info(
            "Computing density within each time point to estimate the time "
            "length scale `ls_time`. Specify `ls_time` to skip this step."
        )
        ls = compute_ls_time(
            nn_distances,
            x,
            cov_func_curry,
            density_estimator_kwargs=density_estimator_kwargs,
        )
        ls *= self.ls_factor_times
        return ls

    def _compute_cov_func(self):
        cov_func_curry = self.cov_func_curry
        ls = self.ls
        ls_time = self.ls_time
        cov_func = compute_cov_func(cov_func_curry, ls, ls_time)
        logger.info("Using covariance function %s.", str(cov_func))
        return cov_func

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
        log_density_func = compute_conditional_mean_times(
            x,
            landmarks,
            pre_transformation,
            log_density_x,
            mu,
            cov_func,
            jitter=jitter,
        )
        self.log_density_func = log_density_func

    def _set_x(self, x, times=None):
        self.x = _validate_time_x(x, times)

    def prepare_inference(self, x, times=None):
        R"""
        Prepares for optimization without performing Bayesian inference.
        This method sets all attributes required for optimization.
        It is not required to call this method manually before calling `fit`.

        Parameters
        ----------
        x : array-like
            The training instances for which the density function will be estimated.
            If 'times' is None, the last column of 'x' is interpreted as the times.
            Shape must be (n_samples, n_features).

        times : array-like, optional
            An array encoding the time points associated with each cell/row in 'x'.
            If provided, it overrides the last column of 'x' as the times.
            Shape must be either (n_samples,) or (n_samples, 1).

        Returns
        -------
        loss_func : function
            The Bayesian loss function that will be minimized during optimization.

        initial_value : array-like
            The initial guess for the optimization process.
        """

        self._set_x(x, times=times)
        self._prepare_attribute("nn_distances")
        self._prepare_attribute("d")
        self._prepare_attribute("mu")
        self._prepare_attribute("ls")
        self._prepare_attribute("ls_time")
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

    def fit(self, x=None, times=None, build_predict=True):
        R"""
        Fit the model from end to end.

        Parameters
        ----------
        x : array-like, optional
            The training instances to estimate density function.
            If 'x' is not provided and 'self.x' is also None, a ValueError is raised.
        times : array-like, optional
            An array encoding the time points associated with each cell/row in 'x'.
            Shape must be either (n_samples,) or (n_samples, 1).
        build_predict : bool, optional
            Whether or not to build the prediction function. Defaults to True.

        Returns
        -------
        self : object
            A fitted instance of this estimator.

        Raises
        ------
        ValueError
            If both 'x' and 'self.x' are None or if 'x' is provided and not equal to 'self.x'.
        """

        if x is not None:
            x = _validate_time_x(x, times)
        if self.x is not None and self.x is not x:
            message = "self.x has been set already, but is not equal to the argument x."
            raise ValueError(message)
        if self.x is None and x is None:
            message = "Required argument x is missing and self.x has not been set."
            raise ValueError(message)
        if x is None:
            x = self.x

        self.prepare_inference(x, times)
        self.run_inference()
        self.process_inference(build_predict=build_predict)
        return self

    @property
    def predict(self):
        R"""
        An instance of the :class:`mellon.Predictor` that predicts the log density at each point in x.

        The instance contains a __call__ method which can be used to predict the log density.
        This instance also supports serialization features which allows for saving
        and loading the predictor state. Refer to mellon.Predictor documentation for more details.

        Note that the last column of the input array `x` should contain the time information.

        Parameters
        ----------
        x : array-like
            The new data to predict, where the last column should contain the time information.

        Returns
        -------
        log_density : array-like
            The log density at each test point in `x`.

        """
        if self.log_density_func is None:
            self._set_log_density_func()
        return self.log_density_func

    def fit_predict(self, x=None, times=None, build_predict=False):
        R"""
        Perform Bayesian inference and return the log density at training points.

        :param x: The training instances to estimate density function.
        :type x: array-like
        :return: log_density_x - The log density at each training point in x.
        """
        if x is not None:
            x = _validate_time_x(x, times)
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

        self.fit(x, build_predict=build_predict)
        return self.log_density_x
