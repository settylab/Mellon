import logging
from .base_model import BaseEstimator, DEFAULT_COV_FUNC
from .inference import (
    compute_dimensionality_transform,
    compute_dimensionality_loss_func,
    compute_log_density_x,
    compute_conditional,
    compute_conditional_explog,
    DEFAULT_N_ITER,
    DEFAULT_INIT_LEARN_RATE,
    DEFAULT_JIT,
    DEFAULT_OPTIMIZER,
)
from .parameters import (
    compute_distances,
    compute_mu,
    compute_initial_dimensionalities,
)
from .util import (
    DEFAULT_JITTER,
    local_dimensionality,
    object_str,
    object_html,
)
from .validation import (
    validate_positive_int,
    validate_float,
    validate_array,
)


logger = logging.getLogger("mellon")


class DimensionalityEstimator(BaseEstimator):
    R"""
    This class provides a non-parametric method for estimating local dimensionality and density.
    It uses Bayesian inference with a Gaussian process prior and a normal distribution for local scaling rates.
    The class caches all intermediate computations as instance variables,
    enabling users to view intermediate results and
    save computational time by passing precomputed values to a new model instance.

    Parameters
    ----------
    cov_func_curry: function or type, optional (default=Matern52)
        A generator for the Gaussian process covariance function. It should be
        a curry function taking one length scale argument
        and returning a covariance function of the form k(x, y) :math:`\rightarrow` float.

    n_landmarks : int, optional (default=5000)
        The number of landmark/inducing points. Only used if a sparse GP is indicated
        through gp_type. If 0 or equal to the number of training points, inducing points
        will not be computed or used.

    rank: int or float, optional (default=0.99)
        The rank of the approximate covariance matrix for the Nyström rank reduction.
        If rank is an int, an :math:`n \times`
        rank matrix :math:`L` is computed such that :math:`L L^\top \approx K`, where `K` is the
        exact :math:`n \times n` covariance matrix. If rank is a float 0.0 :math:`\le` rank
        :math:`\le` 1.0, the rank/size of :math:`L` is selected such that the included eigenvalues
        of the covariance between landmark points account for the specified percentage of the sum
        of eigenvalues. It is ignored if gp_type does not indicate a Nyström rank reduction.

    gp_type : str or GaussianProcessType, optional (default='sparse_cholesky')
        The type of sparcification used for the Gaussian Process:
         - 'full' None-sparse Gaussian Process
         - 'full_nystroem' Sparse GP with Nyström rank reduction without landmarks,
            which lowers the computational complexity.
         - 'sparse_cholesky' Sparse GP using landmarks/inducing points,
            typically employed to enable scalable GP models.
         - 'sparse_nystroem' Sparse GP using landmarks or inducing points,
            along with an improved Nyström rank reduction method.

        The value can be either a string matching one of the above options or an instance of
        the `mellon.util.GaussianProcessType` Enum. If a partial match is found with the
        Enum, a warning will be logged, and the closest match will be used.

    jitter: float, optional (default=1e-6)
        A small amount added to the diagonal of the covariance matrix to ensure
        numerical stability by keeping eigenvalues away from 0.

    optimizer: str, optional (default='L-BFGS-B')
        The optimizer to use for maximum a posteriori density estimation.
        It can be either 'L-BFGS-B', 'adam', or 'advi'.

    n_iter: int, optional (default=100)
        The number of iterations for optimization.

    init_learn_rate: float, optional (default=1)
        The initial learning rate for the optimizer.

    landmarks: array-like or None, optional
        Points used to quantize the data for approximate covariance. If None,
        landmarks are set as k-means centroids
        with k=n_landmarks. If the number of landmarks is greater than or equal
        to the number of training points, this parameter is ignored.

    k: int, optional (default=10)
        The number of nearest neighbor distances to consider.

    distances: array-like or None, optional
        The k nearest neighbor distances at each data point. If None, these
        distances are computed automatically using KDTree (if data dimensionality is < 20)
        or BallTree otherwise.

    d: array-like, optional
        The estimated local intrinsic dimensionality of the data. This is only
        used to initialize the density estimation.
        If None, an empirical estimate is used.

    mu_dim: float or None, optional (default=0)
        The mean of the Gaussian process for log intrinsic dimensionality :math:`\mu_D`.

    mu_dens: float or None, optional
        The mean of the Gaussian process for log density :math:`\mu_\rho`. If None,
        `mu_dens` is set to the 1st percentile of :math:`\text{mle}(\text{nn_distances}, d) - 10`.

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

    Lp : array-like or None
        A matrix such that :math:`L_p L_p^\top = \Sigma_p`, where :math:`\Sigma_p` is the
        covariance matrix of the inducing points (all cells in non-sparse GP).
        Not used when Nyström rank reduction is employed. Defaults to None.

    L: array-like or None, optional
        A matrix such that :math:`L L^\top \approx K`, where :math:`K` is the
        covariance matrix. If None, `L` is computed automatically.

    initial_value: array-like or None, optional
        The initial guess for optimization. If None, an optimized :math:`z` is
        found that minimizes :math:`||Lz + \mu_\cdot - mle|| + ||z||`, where
        :math:`\text{mle}` is the maximum likelihood estimate
        for density initialization and the neighborhood-based local intrinsic
        dimensionality for dimensionality initialization.

    predictor_with_uncertainty : bool
        If set to True, computes the predictor instances `.predict` and `.predict_density`
        with its predictive uncertainty. The uncertainty comes from two sources:

        1) `.predict.mean_covariance`:
            Uncertainty arising from the posterior distribution of the Bayesian inference.
            This component quantifies uncertainties inherent in the model's parameters and structure.
            Available only if `.pre_transformation_std` is defined (e.g., using `optimizer="advi"`),
            which reflects the standard deviation of the latent variables before transformation.

        2) `.predict.covariance`:
            Uncertainty for out-of-bag states originating from the compressed function representation
            in the Gaussian Process. Specifically, this uncertainty corresponds to locations that are
            not inducing points of the Gaussian Process and represents the covariance of the
            conditional normal distribution.

    jit: bool, optional (default=False)
        If True, use JAX's just-in-time compilation for loss and its gradient during optimization.

    check_rank : bool
        Weather to check if landmarks allow sufficient complexity by checking the approximate
        rank of the covariance matrix. This only applies to the non-Nyström gp_types.
        If set to None the rank check is only performed if n_landmarks >= n_samples/10.
        Defaults to None.
    """

    def __init__(
        self,
        cov_func_curry=DEFAULT_COV_FUNC,
        n_landmarks=None,
        rank=None,
        gp_type=None,
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
        Lp=None,
        L=None,
        initial_value=None,
        predictor_with_uncertainty=False,
        jit=DEFAULT_JIT,
        check_rank=None,
    ):
        super().__init__(
            cov_func_curry=cov_func_curry,
            n_landmarks=n_landmarks,
            rank=rank,
            gp_type=gp_type,
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
            Lp=Lp,
            L=L,
            initial_value=initial_value,
            predictor_with_uncertainty=predictor_with_uncertainty,
            jit=jit,
            check_rank=check_rank,
        )
        self.k = validate_positive_int(k, "k")
        self.mu_dim = validate_float(mu_dim, "mu_dim")
        self.mu_dens = validate_float(mu_dens, "mu_dens", optional=True)
        self.distances = validate_array(distances, "distances", optional=True)
        self.transform = None
        self.loss_func = None
        self.opt_state = None
        self.losses = None
        self.pre_transformation = None
        self.pre_transformation_std = None
        self.local_dim_x = None
        self.log_density_x = None
        self.local_dim_func = None
        self.log_density_func = None

    def __repr__(self):
        name = self.__class__.__name__
        landmarks = object_str(self.landmarks, ["landmarks", "dims"])
        Lp = object_str(self.Lp, ["landmarks", "landmarks"])
        L = object_str(self.L, ["cells", "ranks"])
        nn_distances = object_str(self.nn_distances, ["cells"])
        d = object_str(self.d, ["cells"])
        initial_value = object_str(self.initial_value, ["functions", "ranks"])
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
            f"\n    mu_dens={self.mu_dens},"
            f"\n    mu_dim={self.mu_dim},"
            f"\n    n_landmarks={self.n_landmarks},"
            f"\n    nn_distances={nn_distances},"
            f"\n    optimizer={self.optimizer},"
            f"\n    predictor_with_uncertainty={self.predictor_with_uncertainty},"
            f"\n    rank={self.rank},"
            "\n)"
        )
        return string

    def _repr_html_(self):
        """
        Generate an HTML representation for the DimensionalityEstimator subclass for display in Jupyter.
        """

        # Header with class name and purpose
        header = f"""
        <h2>Dimensionality Estimator: {self.__class__.__name__}</h2>
        <p><em>A non-parametric method for estimating local dimensionality and density using Gaussian Processes.</em></p>
        """

        # Core attributes as a list
        core_attributes = f"""
        <h3>Core Attributes</h3>
        <ul>
            <li><strong>Covariance Function:</strong> {object_html(self.cov_func or 'Not Set')}</li>
            <li><strong>Optimizer:</strong> {self.optimizer}</li>
            <li><strong>Number of Landmarks:</strong> {self.n_landmarks or 'Not Set'}</li>
            <li><strong>Gaussian Process Type:</strong> {self.gp_type or 'Not Set'}</li>
            <li><strong>Rank Reduction:</strong> {self.rank or 'Not Applicable'}</li>
            <li><strong>Predictor with Uncertainty:</strong> {'Yes' if self.predictor_with_uncertainty else 'No'}</li>
        </ul>
        """

        # Parameters as a table
        parameters = {
            "Jitter": self.jitter,
            "Mean Dimensionality (μ_dim)": self.mu_dim,
            "Mean Density (μ_dens)": self.mu_dens or "Not Set",
            "Length Scale (ls)": self.ls or "Not Set",
            "Length Scale Factor": self.ls_factor,
            "Intrinsic Dimensionality (d)": self.d,
            "Number of Nearest Neighbors (k)": self.k,
            "Distances": self.distances,
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
            "<p style='color:green;'><strong>Predictors:</strong> Available</p>"
            if self.local_dim_func and self.log_density_func
            else "<p style='color:red;'><strong>Predictors:</strong> Not Yet Computed</p>"
        )

        # Predictor details if available
        predictor_details = ""
        if self.local_dim_func and self.log_density_func:
            predictor_details = f"""
            <h3>Predictor Details</h3>
            <p><strong>Dimensionality Predictor Instance:</strong> {object_html(self.local_dim_func)}</p>
            <p><strong>Density Predictor Instance:</strong> {object_html(self.log_density_func)}</p>
            """

        # Combine all sections
        return (
            header
            + core_attributes
            + parameters_table
            + predictor_status
            + predictor_details
        )

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
        pre_transformation = self.pre_transformation[0, :]
        pre_transformation_std = self.pre_transformation_std
        if pre_transformation_std is not None:
            pre_transformation_std = pre_transformation_std[0, :]
        local_dim_x = self.local_dim_x
        mu = self.mu_dim
        cov_func = self.cov_func
        L = self.L
        Lp = self.Lp
        jitter = self.jitter
        with_uncertainty = self.predictor_with_uncertainty
        logger.info("Computing predictive dimensionality function.")
        log_dim_func = compute_conditional_explog(
            x,
            landmarks,
            pre_transformation,
            pre_transformation_std,
            local_dim_x,
            mu,
            cov_func,
            L,
            Lp,
            sigma=None,
            jitter=jitter,
            y_is_mean=True,
            with_uncertainty=with_uncertainty,
        )
        self.local_dim_func = log_dim_func

    def _set_log_density_func(self):
        x = self.x
        landmarks = self.landmarks
        pre_transformation = self.pre_transformation[1, :]
        pre_transformation_std = self.pre_transformation_std
        if pre_transformation_std is not None:
            pre_transformation_std = pre_transformation_std[1, :]
        log_density_x = self.log_density_x
        mu = self.mu_dens
        cov_func = self.cov_func
        L = self.L
        Lp = self.Lp
        jitter = self.jitter
        with_uncertainty = self.predictor_with_uncertainty
        logger.info("Computing predictive density function.")
        log_density_func = compute_conditional(
            x,
            landmarks,
            pre_transformation,
            pre_transformation_std,
            log_density_x,
            mu,
            cov_func,
            L,
            Lp,
            sigma=None,
            jitter=jitter,
            y_is_mean=True,
            with_uncertainty=with_uncertainty,
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
        if x is None:
            x = self.x
            if self.x is None:
                message = "Required argument x is missing and self.x has not been set."
                raise ValueError(message)
        else:
            if self.x is not None and self.x is not x:
                message = (
                    "self.x has been set already, but is not equal to the argument x."
                )
                raise ValueError(message)

        x = self.set_x(x)
        self._prepare_attribute("n_landmarks")
        self._prepare_attribute("rank")
        self._prepare_attribute("gp_type")
        self.validate_parameter()
        self._prepare_attribute("distances")
        self._prepare_attribute("nn_distances")
        self._prepare_attribute("d")
        self._prepare_attribute("mu_dens")
        self._prepare_attribute("ls")
        self._prepare_attribute("cov_func")
        self._prepare_attribute("landmarks")
        self._prepare_attribute("Lp")
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
        """
        Trains the model from start to finish. This includes preparing for inference, running inference,
        and processing the inference results.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features), default=None
            The training instances where `n_samples` is the number of samples and `n_features`
            is the number of features.

        build_predict : bool, default=True
            Whether or not to construct the prediction function after training.

        Returns
        -------
        self
            A fitted instance of this estimator.
        """

        self.prepare_inference(x)
        self.run_inference()
        self.process_inference(build_predict=build_predict)
        return self

    @property
    def predict_density(self):
        """
        A property that returns an instance of the :class:`mellon.Predictor` class. This predictor can
        be used to predict the log density for new data points by calling the instance like a function.

        The predictor instance also supports serialization features, which allow for saving and loading
        the predictor's state. For more details, refer to the :class:`mellon.Predictor` documentation.

        Returns
        -------
        mellon.Predictor
            A predictor instance that computes the log density at each new data point.

        Example
        -------

        >>> log_density = model.predict_density(Xnew)

        """
        if self.log_density_func is None:
            self._set_log_density_func()
        return self.log_density_func

    @property
    def predict(self):
        """
        A property that returns an instance of the :class:`mellon.base_predictor.ExpPredictor` class.
        This predictor can be used to predict the dimensionality for new data points by calling
        the instance like a function.

        The predictor instance also supports serialization features, which allow for saving and loading
        the predictor's state. For more details, refer to the :class:`mellon.base_predictor.ExpPredictor`
        documentation.

        Returns
        -------
        mellon.base_predictor.ExpPredictor
            A predictor instance that computes the dimensionality at each new data point.

        Example
        -------

        >>> log_density = model.predict(Xnew)

        """
        if self.local_dim_func is None:
            self._set_local_dim_func()
        return self.local_dim_func

    def fit_predict(self, x=None, build_predict=False):
        """
        Trains the model using the provided training data, and then makes predictions
        on the trained data points. This function performs Bayesian inference to compute
        the local dimensionality and returns the computed local dimensionality at each
        training point.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features), default=None
            The training instances to estimate the local dimensionality function, where
            `n_samples` is the number of samples and `n_features` is the number of features.
            Each sample is an array of features representing a point in the feature space.

        build_predict : bool, default=False
            Whether or not to build the prediction function after training.

        Returns
        -------
        array-like of shape (n_samples,)
            The local dimensionality at each training point in `x`.

        Raises
        ------
        ValueError
            If the argument `x` does not match `self.x` which was already set in a previous operation.
        """
        if self.x is not None and x is not None and self.x is not x:
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
            x = validate_array(x, "x")

        self.fit(x, build_predict=build_predict)
        return self.local_dim_x
