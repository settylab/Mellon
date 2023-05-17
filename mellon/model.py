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
    DEFAULT_N_LANDMARKS,
)
from .derivatives import (
    gradient,
    hessian,
    hessian_log_determinant,
)
from .util import (
    DEFAULT_JITTER,
    vector_map,
    Log,
    local_dimensionality,
)


DEFAULT_D_METHOD = "embedding"

logger = Log()


class DensityEstimator(BaseEstimator):
    R"""
    A non-parametric density estimator.
    DensityEstimator performs Bayesian inference with a Gaussian process prior and Nearest
    Neighbor likelihood. All intermediate computations are cached as instance variables, so
    the user can view intermediate results and save computation time by passing precomputed
    values as arguments to a new model.

    :param cov_func_curry: The generator of the Gaussian process covariance function.
        Must be a curry that takes one length scale argument and returns a
        covariance function of the form k(x, y) :math:`\rightarrow` float.
        Defaults to the type Matern52.
    :type cov_func_curry: function or type
    :param n_landmarks: The number of landmark points. If less than 1 or greater than or
        equal to the number of training points, does not compute or use inducing points.
        Defaults to 5000.
    :type n_landmarks: int
    :param rank: The rank of the approximate covariance matrix.
        If rank is an int, an :math:`n \times` rank matrix
        :math:`L` is computed such that :math:`L L^\top \approx K`, the exact
        :math:`n \times n` covariance matrix.
        If rank is a float 0.0 :math:`\le` rank :math:`\le` 1.0, the rank/size
        of :math:`L` is selected such that the included eigenvalues of the covariance
        between landmark points account for the specified percentage of the
        sum of eigenvalues. Defaults to 0.99.
    :type rank: int or float
    :param method: Explicitly specifies whether rank is to be interpreted as a
        fixed number of eigenvectors or a percent of eigenvalues to include
        in the low rank approximation. Supports 'fixed', 'percent', or 'auto'.
        If 'auto', interprets rank as a fixed number of eigenvectors if it is
        an int and interprets rank as a percent of eigenvalues if it is a float.
        Provided for explictness and to clarify the ambiguous case of 1 vs 1.0.
        Defaults to 'auto'.
    :type method: str
    :param d_method: Method to compute intrinsic dimensionality of the data.
        Implemended options are
        * 'embedding' use the embedding dimension `x.shape[1]`
        * 'fractal' use the average fractal dimension (experimental)
        Defaults to 'embedding'.
    :type d_method: str
    :param jitter: A small amount to add to the diagonal of the covariance
        matrix to bind eigenvalues numerically away from 0 ensuring numerical
        stabilitity. Defaults to 1e-6.
    :type jitter: float
    :param optimizer: Select optimizer 'L-BFGS-B' or stochastic optimizer 'adam'
        for the maximum a posteriori density estimation. Defaults to 'L-BFGS-B'.
    :type optimizer: str
    :param n_iter: The number of optimization iterations. Defaults to 100.
    :type n_iter: int
    :param init_learn_rate: The initial learn rate. Defaults to 1.
    :type init_learn_rate: float
    :param landmarks: The points to quantize the data for the approximate covariance. If None,
        landmarks are set as k-means centroids with k=n_landmarks. Ignored if n_landmarks
        is greater than or equal to the number of training points. Defaults to None.
    :type landmarks: array-like or None
    :param nn_distances: The nearest neighbor distances at each
        data point. If None, computes the nearest neighbor distances automatically, with
        a KDTree if the dimensionality of the data is less than 20, or a BallTree otherwise.
        Defaults to None.
    :type nn_distances: array-like or None
    :param d: The intrinsic dimensionality of the data, i.e., the dimansionality of
        the embedded manifold.
        If None, sets d to the size of axis 1
        of the training data points. Defaults to None.
    :type d: int, array-like or None
    :param mu: The mean :math:`\mu` of the Gaussian process. If None, sets
        :math:`\mu` to the 1th percentile
        of :math:`\text{mle}(\text{nn_distances}, d) - 10`, where
        :math:`\text{mle} = \log(\text{gamma}(d/2 + 1))
        - (d/2) \cdot \log(\pi) - d \cdot \log(\text{nn_distances})`. Defaults to None.
    :type mu: float or None
    :param ls: The length scale of the Gaussian process covariance function. If None,
        sets ls to the geometric mean of the nearest neighbor distances times a constant.
        If cov_func is supplied explictly, ls has no effect. Defaults to None.
    :type ls: float or None
    :param cov_func: The Gaussian process covariance function of the form
        k(x, y) :math:`\rightarrow` float. If None, automatically generates the covariance
        function cov_func = cov_func_curry(ls). Defaults to None.
    :type cov_func: function or None
    :param L: A matrix such that :math:`L L^\top \approx K`, where :math:`K` is the covariance matrix.
        If None, automatically computes L. Defaults to None.
    :type L: array-like or None
    :param initial_value: The initial guess for optimization. If None, finds
        :math:`z` that minimizes :math:`||Lz + \mu - mle|| + ||z||`, where
        :math:`\text{mle} = \log(\text{gamma}(d/2 + 1)) - (d/2) \cdot \log(\pi)
        - d \cdot \log(\text{nn_distances})`,
        where :math:`d` is the intrinsic dimensionality of the data. Defaults to None.
    :type initial_value: array-like or None
    :param jit: Use jax just in time compilation for loss and its gradient
        during optimization. Defaults to False.
    :type jit: bool
    :ivar cov_func_curry: The generator of the Gaussian process covariance function.
    :ivar n_landmarks: The number of landmark points.
    :ivar rank: The rank of approximate covariance matrix or percentage of
        eigenvalues included in approximate covariance matrix.
    :ivar method: The method to interpret the rank as a fixed number of eigenvectors
        or a percentage of eigenvalues.
    :ivar d_method: The method to determin intrinsic dimensionality.
    :ivar jitter: A small amount added to the diagonal of the covariance matrix
        for numerical stability.
    :ivar n_iter: The number of optimization iterations if adam optimizer is used.
    :ivar init_learn_rate: The initial learn rate when adam optimizer is used.
    :ivar landmarks: The points to quantize the data.
    :ivar nn_distances: The nearest neighbor distances for each data point.
    :ivar d: The local dimensionality of the data.
    :ivar mu: The Gaussian process mean :math:`\mu`.
    :ivar ls: The Gaussian process covariance function length scale.
    :ivar ls_factor: Factor to scale the automatically selected length scale.
        Defaults to 1.
    :ivar cov_func: The Gaussian process covariance function.
    :ivar L: A matrix such that :math:`L L^\top \approx K`, where :math:`K` is the covariance matrix.
    :ivar initial_value: The initial guess for Maximum A Posteriori optimization.
    :ivar optimizer: Optimizer for the maximum a posteriori density estimation.
    :ivar x: The training data.
    :ivar transform: A function
        :math:`z \sim \text{Normal}(0, I) \rightarrow \text{Normal}(\mu, K')`.
        Used to map the latent representation to the log-density on the
        training data.
    :ivar loss_func: The Bayesian loss function.
    :ivar pre_transformation: The optimized parameters :math:`z \sim \text{Normal}(0, I)` before
        transformation to :math:`\text{Normal}(\mu, K')`, where :math:`I` is the identity matrix
        and :math:`K'` is the approximate covariance matrix.
    :ivar opt_state: The final state the optimizer.
    :ivar losses: The history of losses throughout training of adam or final
        loss of L-BFGS-B.
    :ivar log_density_x: The log density at the training points.
    :ivar log_density_func: A function that computes the log density at arbitrary prediction points.
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
            landmarks=landmarks,
            nn_distances=nn_distances,
            mu=mu,
            ls=ls,
            ls_factor=ls_factor,
            cov_func=cov_func,
            L=L,
        )
        self.method = method
        self.d_method = d_method
        self.optimizer = optimizer
        self.n_iter = n_iter
        self.init_learn_rate = init_learn_rate
        self.initial_value = initial_value
        self.d = d
        self.transform = None
        self.loss_func = None
        self.opt_state = None
        self.losses = None
        self.pre_transformation = None
        self.log_density_x = None
        self.log_density_func = None
        self.jit = jit

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
            self.pre_transformation = pre_transformation
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

        self.prepare_inference(x)
        self.run_inference()
        self.process_inference(build_predict=build_predict)
        return self

    def predict(self, x):
        R"""
        Predict the log density at each point in x.

        :param x: The new data to predict.
        :type x: array-like
        :return: log_density - The log density at each test point in x.
        :rtype: array-like
        """
        if self.log_density_func is None:
            self._set_log_density_func()
        return self.log_density_func(x)

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

        self.fit(x, build_predict=build_predict)
        return self.log_density_x


class FunctionEstimator(BaseEstimator):
    R"""
    Uses a conditional normal distribution to smoothen and extend a function
    on all cell states using the Mellon abstractions.

    :param cov_func_curry: The generator of the Gaussian process covariance function.
        Must be a curry that takes one length scale argument and returns a
        covariance function of the form k(x, y) :math:`\rightarrow` float.
        Defaults to the type Matern52.
    :type cov_func_curry: function or type
    :param n_landmarks: The number of landmark points. If less than 1 or greater than or
        equal to the number of training points, does not compute or use inducing points.
        Defaults to 5000.
    :type n_landmarks: int
    :type method: str
    :param jitter: A small amount to add to the diagonal of the covariance
        matrix to bind eigenvalues numerically away from 0 ensuring numerical
        stabilitity. Defaults to 1e-6.
    :type jitter: float
    :param landmarks: The points to quantize the data for the approximate covariance. If None,
        landmarks are set as k-means centroids with k=n_landmarks. Ignored if n_landmarks
        is greater than or equal to the number of training points. Defaults to None.
    :type landmarks: array-like or None
    :param nn_distances: The nearest neighbor distances at each
        data point. If None, computes the nearest neighbor distances automatically, with
        a KDTree if the dimensionality of the data is less than 20, or a BallTree otherwise.
        Defaults to None.
    :type nn_distances: array-like or None
    :param mu: The mean of the Gaussian process :math:`\mu`. Defaults to 0.
    :type mu: float or None
    :param ls: The length scale of the Gaussian process covariance function. If None,
        sets ls to the geometric mean of the nearest neighbor distances times a constant.
        If cov_func is supplied explictly, ls has no effect. Defaults to None.
    :type ls: float or None
    :param cov_func: The Gaussian process covariance function of the form
        k(x, y) :math:`\rightarrow` float. If None, automatically generates the covariance
        function cov_func = cov_func_curry(ls). Defaults to None.
    :type cov_func: function or None
    :param sigma: The white moise standard deviation. Defaults to 0.
    :type sigma: float
    :ivar n_landmarks: The number of landmark points.
    :ivar jitter: A small amount added to the diagonal of the covariance matrix
        for numerical stability.
    :ivar landmarks: The points to quantize the data.
    :ivar nn_distances: The nearest neighbor distances for each data point.
    :ivar mu: The Gaussian process mean :math:`\mu`.
    :ivar ls: The Gaussian process covariance function length scale.
    :ivar ls_factor: Factor to scale the automatically selected length scale.
        Defaults to 1.
    :ivar cov_func: The Gaussian process covariance function.
    :ivar sigma: White noise standard deviation.
    :ivar x: The cell states.
    :ivar y: Function values on cell states.
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
        )
        self.sigma = sigma

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
        if self.x is not None and self.x is not x:
            message = "self.x has been set already, but is not equal to the argument x."
            raise ValueError(message)
        if self.x is None and x is None:
            message = "Required argument x is missing and self.x has not been set."
            raise ValueError(message)
        if x is None:
            x = self.x
        if y is None:
            message = "Required argument y is missing."
            raise ValueError(message)
        landmarks = self.landmarks
        pre_transformation = self.pre_transformation
        mu = self.mu
        cov_func = self.cov_func
        sigma = self.sigma
        jitter = self.jitter
        conditional = compute_conditional_mean(
            x,
            landmarks,
            pre_transformation,
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

    def predict(self, x):
        R"""
        Predict the function at each point in x.

        :param x: The new data to predict.
        :type x: array-like
        :return: condition_mean - The conditional mean function value at each test point in x.
        :rtype: array-like
        """
        return self.conditional(x)

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

        if Xnew is None:
            Xnew = x

        self.prepare_inference(x)

        landmarks = self.landmarks
        mu = self.mu
        cov_func = self.cov_func
        sigma = self.sigma
        jitter = self.jitter

        conditional = compute_conditional_mean_y(
            x,
            landmarks,
            Xnew,
            mu,
            cov_func,
            sigma,
            jitter=jitter,
        )

        return vector_map(conditional, Y)


class DimensionalityEstimator(BaseEstimator):
    R"""
    A non-parametric estimator for local dimensionality and density.
    DimensionalityEstimator performs Bayesian inference with a Gaussian process prior and
    a normal distribution for local scaling rates.
    All intermediate computations are cached as instance variables, so
    the user can view intermediate results and save computation time by passing precomputed
    values as arguments to a new model.

    :param cov_func_curry: The generator of the Gaussian process covariance function.
        Must be a curry that takes one length scale argument and returns a
        covariance function of the form k(x, y) :math:`\rightarrow` float.
        Defaults to the type Matern52.
    :type cov_func_curry: function or type
    :param n_landmarks: The number of landmark points. If less than 1 or greater than or
        equal to the number of training points, does not compute or use inducing points.
        Defaults to 5000.
    :type n_landmarks: int
    :param rank: The rank of the approximate covariance matrix.
        If rank is an int, an :math:`n \times` rank matrix
        :math:`L` is computed such that :math:`L L^\top \approx K`, the exact
        :math:`n \times n` covariance matrix.
        If rank is a float 0.0 :math:`\le` rank :math:`\le` 1.0, the rank/size
        of :math:`L` is selected such that the included eigenvalues of the covariance
        between landmark points account for the specified percentage of the
        sum of eigenvalues. Defaults to 0.99.
    :type rank: int or float
    :param method: Explicitly specifies whether rank is to be interpreted as a
        fixed number of eigenvectors or a percent of eigenvalues to include
        in the low rank approximation. Supports 'fixed', 'percent', or 'auto'.
        If 'auto', interprets rank as a fixed number of eigenvectors if it is
        an int and interprets rank as a percent of eigenvalues if it is a float.
        Provided for explictness and to clarify the ambiguous case of 1 vs 1.0.
        Defaults to 'auto'.
    :type method: str
    :param jitter: A small amount to add to the diagonal of the covariance
        matrix to bind eigenvalues numerically away from 0 ensuring numerical
        stabilitity. Defaults to 1e-6.
    :type jitter: float
    :param optimizer: Select optimizer 'L-BFGS-B' or stochastic optimizer 'adam'
        for the maximum a posteriori density estimation. Defaults to 'L-BFGS-B'.
    :type optimizer: str
    :param n_iter: The number of optimization iterations. Defaults to 100.
    :type n_iter: int
    :param init_learn_rate: The initial learn rate. Defaults to 1.
    :type init_learn_rate: float
    :param landmarks: The points to quantize the data for the approximate covariance. If None,
        landmarks are set as k-means centroids with k=n_landmarks. Ignored if n_landmarks
        is greater than or equal to the number of training points. Defaults to None.
    :type landmarks: array-like or None
    :param k: The number of nearest neighbor distances to consider. Defaults to 10.
    :type k: int
    :param distances: The k nearest neighbor distances at each
        data point. If None, computes the nearest neighbor distances automatically, with
        a KDTree if the dimensionality of the data is less than 20, or a BallTree otherwise.
        Defaults to None.
    :type distances: array-like or None
    :param d: The estimated local intrinsic dimensionality of the data.
        This is only used to initialize the density estimation.
        If None, sets d to the emperical estimae.
    :type d: array-like
    :param mu_dim: The mean of the Gaussian process for log intrinsic
        dimensionality :math:`\mu_D. Default is 0.
    :type mu_dim: float or None
    :param mu_dens: The mean of the Gaussian process for log density
        :math:`\mu_\rho`. If None,
        sets mu_dens to the 1th percentile
        of :math:`\text{mle}(\text{nn_distances}, d) - 10`, where
        :math:`\text{mle} = \log(\text{gamma}(d/2 + 1))
        - (d/2) \cdot \log(\pi) - d \cdot \log(\text{nn_distances})`. Defaults to None.
    :type mu_dens: float or None
    :param ls: The length scale of the Gaussian process covariance function. If None,
        sets ls to the geometric mean of the nearest neighbor distances times a constant.
        If cov_func is supplied explictly, ls has no effect. Defaults to None.
    :type ls: float or None
    :param cov_func: The Gaussian process covariance function of the form
        k(x, y) :math:`\rightarrow` float. If None, automatically generates the covariance
        function cov_func = cov_func_curry(ls). Defaults to None.
    :type cov_func: function or None
    :param L: A matrix such that :math:`L L^\top \approx K`, where :math:`K` is the covariance matrix.
        If None, automatically computes L. Defaults to None.
    :type L: array-like or None
    :param initial_value: The initial guess for optimization. If None, finds
        :math:`z` that minimizes :math:`||Lz + \mu_\cdot - mle|| + ||z||`, where
        :math:`\text{mle}` is the maximum likelyhood estimate for the denisty
        initialization and the neighborhood based local intrinstic dimensionality
        for the dimensionality initializatuion.
    :type initial_value: array-like or None
    :param jit: Use jax just in time compilation for loss and its gradient
        during optimization. Defaults to False.
    :type jit: bool
    :ivar cov_func_curry: The generator of the Gaussian process covariance function.
    :ivar n_landmarks: The number of landmark points.
    :ivar rank: The rank of approximate covariance matrix or percentage of
        eigenvalues included in approximate covariance matrix.
    :ivar method: The method to interpret the rank as a fixed number of eigenvectors
        or a percentage of eigenvalues.
    :ivar jitter: A small amount added to the diagonal of the covariance matrix
        for numerical stability.
    :ivar n_iter: The number of optimization iterations if adam optimizer is used.
    :ivar init_learn_rate: The initial learn rate when adam optimizer is used.
    :ivar landmarks: The points to quantize the data.
    :ivar nn_distances: The nearest neighbor distances for each data point.
    :ivar mu_dim: The Gaussian process mean :math:`\mu_D`.
    :ivar mu_dens: The Gaussian process mean :math:`\mu_\rho`.
    :ivar ls: The Gaussian process covariance function length scale.
    :ivar ls_factor: Factor to scale the automatically selected length scale.
        Defaults to 1.
    :ivar cov_func: The Gaussian process covariance function.
    :ivar L: A matrix such that :math:`L L^\top \approx K`, where :math:`K` is the covariance matrix.
    :ivar initial_value: The initial guess for Maximum A Posteriori optimization.
    :ivar optimizer: Optimizer for the maximum a posteriori density estimation.
    :ivar x: The training data.
    :ivar transform: A function
        :math:`z \sim \text{Normal}(0, I) \rightarrow \text{Normal}(\mu_\cdot, K')`.
        Used to map the latent representation to the log-density on the
        training data.
    :ivar loss_func: The Bayesian loss function.
    :ivar pre_transformation: The optimized parameters :math:`z \sim \text{Normal}(0, I)` before
        transformation to :math:`\text{Normal}(\mu_\cdot, K')`, where :math:`I` is the identity matrix
        and :math:`K'` is the approximate covariance matrix.
    :ivar opt_state: The final state the optimizer.
    :ivar losses: The history of losses throughout training of adam or final
        loss of L-BFGS-B.
    :ivar local_dim_x: The local intrinsic dimensionality at the training points.
    :ivar log_density_x: The log density with variing units at the training
        points. Density indicates the number of cells per volume in state
        space. Since the intrinsic dimensionality of the volume changes, the resulting
        density unit varies.
    :ivar local_dim_func: A function that computes the local intrinsic dimensionality
        at arbitrary prediction points.
    :ivar log_density_func: A function that computes the log density with
        variing units at arbitrary prediction points.
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
            jitter=jitter,
            landmarks=landmarks,
            nn_distances=None,
            mu=mu_dens,
            ls=ls,
            ls_factor=ls_factor,
            cov_func=cov_func,
            L=L,
        )
        self.k = k
        self.d = d
        self.mu_dim = mu_dim
        self.mu_dens = mu_dens
        self.method = method
        self.distances = distances
        self.optimizer = optimizer
        self.n_iter = n_iter
        self.init_learn_rate = init_learn_rate
        self.initial_value = initial_value
        self.transform = None
        self.loss_func = None
        self.opt_state = None
        self.losses = None
        self.pre_transformation = None
        self.local_dim_x = None
        self.log_density_x = None
        self.local_dim_func = None
        self.log_density_func = None
        self.jit = jit

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

        self.prepare_inference(x)
        self.run_inference()
        self.process_inference(build_predict=build_predict)
        return self

    def predict_density(self, x):
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
        return self.log_density_func(x)

    def predict(self, x):
        R"""
        Predict the dimensionality at each point in x.
        Alias for predict_dimensionality().

        :param x: The new data to predict.
        :type x: array-like
        :return: dimensionality - The dimensionality at each test point in x.
        :rtype: array-like
        """
        if self.local_dim_func is None:
            self._set_local_dim_func()
        return self.local_dim_func(x)

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

        self.fit(x, build_predict=build_predict)
        return self.local_dim_x

    def gradient_density(self, x, jit=True):
        R"""
        Conputes the gradient of the predictive log-density function for each line in x.

        :param x: Data points.
        :type x: array-like
        :param jit: Use jax just in time compilation. Defaults to True.
        :type jit: bool
        :return: gradiants - The gradient of function at each point in x.
            gradients.shape == x.shape
        :rtype: array-like
        """
        return gradient(self.predict_density, x, jit=jit)

    def hessian_density(self, x, jit=True):
        R"""
        Conputes the hessian of the predictive log-density function for each line in x.

        :param x: Data points.
        :type x: array-like
        :param jit: Use jax just in time compilation. Defaults to True.
        :type jit: bool
        :return: hessians - The hessian matrix of function at each point in x.
            hessians.shape == X.shape + X.shape[1:]
        :rtype: array-like
        """
        return hessian(self.predict_density, x, jit=jit)

    def hessian_log_determinant_density(self, x, jit=True):
        R"""
        Conputes the logarirhm of the determinat of the predictive density function for
        each line in x.

        :param x: Data points.
        :type x: array-like
        :param jit: Use jax just in time compilation. Defaults to True.
        :type jit: bool
        :return: signs, log_determinants - The sign of the determinant
            at each point x and the logarithm of its absolute value.
            signs.shape == log_determinants.shape == x.shape[0]
        :rtype: array-like, array-like
        """
        return hessian_log_determinant(self.predict_density, x, jit=jit)
