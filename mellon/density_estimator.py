import logging
from .base_model import BaseEstimator, DEFAULT_COV_FUNC
from .inference import (
    compute_transform,
    compute_loss_func,
    compute_log_density_x,
    compute_conditional,
    DEFAULT_N_ITER,
    DEFAULT_INIT_LEARN_RATE,
    DEFAULT_JIT,
    DEFAULT_OPTIMIZER,
)
from .parameters import (
    compute_d,
    compute_d_factal,
    compute_mu,
    compute_initial_value,
)
from .util import (
    DEFAULT_JITTER,
    object_html,
)
from .validation import (
    validate_string,
    validate_array,
)


DEFAULT_D_METHOD = "embedding"

logger = logging.getLogger("mellon")


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
        The number of landmark/inducing points. Only used if a sparse GP is indicated
        through gp_type. If 0 or equal to the number of training points, inducing points
        will not be computed or used. Defaults to 5000.

    rank : int or float
        The rank of the approximate covariance matrix for the Nyström rank reduction.
        If rank is an int, an :math:`n \times`
        rank matrix :math:`L` is computed such that :math:`L L^\top \approx K`, where `K` is the
        exact :math:`n \times n` covariance matrix. If rank is a float 0.0 :math:`\le` rank
        :math:`\le` 1.0, the rank/size of :math:`L` is selected such that the included eigenvalues
        of the covariance between landmark points account for the specified percentage of the sum
        of eigenvalues. It is ignored if gp_type does not indicate a Nyström rank reduction.
        Defaults to 0.99.

    gp_type : str or GaussianProcessType
        The type of sparcification used for the Gaussian Process
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
        Defaults to 'sparse_cholesky'.

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

    L : array-like or None
        A matrix such that :math:`L L^\top \approx K`, where :math:`K` is the covariance matrix.
        If None, `L` is computed automatically. Defaults to None.

    initial_value : array-like or None
        The initial guess for optimization. If None, the value :math:`z` that minimizes
        :math:`||Lz + \mu - mle|| + ||z||` is found, where :math:`\text{mle} = \log(\text{gamma}(d/2 + 1))
        - (d/2) \cdot \log(\pi) - d \cdot \log(\text{nn_distances})` and :math:`d` is the intrinsic
        dimensionality of the data. Defaults to None.

    predictor_with_uncertainty : bool
        If set to True, computes the predictor instance `.predict` with its predictive uncertainty.
        The uncertainty comes from two sources:

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

    jit : bool
        Use jax just-in-time compilation for loss and its gradient during optimization.
        Defaults to False.

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
            jitter=jitter,
            gp_type=gp_type,
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
            Lp=Lp,
            L=L,
            initial_value=initial_value,
            predictor_with_uncertainty=predictor_with_uncertainty,
            jit=jit,
            check_rank=check_rank,
        )
        self.d_method = validate_string(
            d_method, "d_method", choices={"fractal", "embedding"}
        )
        self.transform = None
        self.loss_func = None
        self.opt_state = None
        self.losses = None
        self.pre_transformation = None
        self.pre_transformation_std = None
        self.log_density_x = None
        self.log_density_func = None

    def _repr_html_(self):
        """
        Generate an HTML representation for the DensityEstimator subclass for display in Jupyter.
        """

        # Header with class name and description
        header = f"""
        <h2>Density Estimator</h2>
        <p><em>A non-parametric density estimation model using Gaussian Processes and Nearest Neighbor Distance Distribution.</em></p>
        """

        # Core attributes as a list
        core_attributes = f"""
        <h3>Core Attributes</h3>
        <ul>
            <li><strong>Covariance Function:</strong> {object_html(self.cov_func or 'Not Set')}</li>
            <li><strong>Optimizer:</strong> {self.optimizer}</li>
            <li><strong>Number of Landmarks:</strong> {self.n_landmarks or 'Not Set'}</li>
            <li><strong>Gaussian Process Type:</strong> {self.gp_type or 'Not Set'}</li>
            <li><strong>Dimensionality Method:</strong> {self.d_method}</li>
        </ul>
        """

        # Table of model parameters with compact columns
        parameters = {
            "Jitter": self.jitter,
            "Mean (μ)": self.mu or "Not Set",
            "Length Scale (ls)": self.ls or "Not Set",
            "Length-Scale Factor": self.ls_factor,
            "Dimensionality (d)": self.d or "Not Set",
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
            if self.log_density_func
            else "<p style='color:red;'><strong>Predictor:</strong> Not Yet Computed</p>"
        )

        # Details about the predictor if available
        predictor_details = ""
        if self.log_density_func:
            predictor_details = f"""
            <h3>Predictor Details</h3>
            <p><strong>Predictor Instance:</strong> {object_html(self.log_density_func)}</p>
            """

        # Combine all sections
        return (
            header
            + core_attributes
            + parameters_table
            + predictor_status
            + predictor_details
        )

    def _compute_d(self):
        x = self.x
        if self.d_method == "fractal":
            d = compute_d_factal(x)
            logger.info(f"Using d={d}.")
        else:
            d = compute_d(x)
            logger.info(
                f"Using embedding dimensionality d={d}. "
                'Use d_method="fractal" to enable effective density normalization.'
            )
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
        pre_transformation_std = self.pre_transformation_std
        log_density_x = self.log_density_x
        mu = self.mu
        cov_func = self.cov_func
        L = self.L
        Lp = self.Lp
        jitter = self.jitter
        with_uncertainty = self.predictor_with_uncertainty
        logger.info("Computing predictive function.")
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
        self._prepare_attribute("nn_distances")
        self._prepare_attribute("d")
        self._prepare_attribute("mu")
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
            self.pre_transformation = validate_array(
                pre_transformation, "pre_transformation"
            )
        self._set_log_density_x()
        if build_predict:
            self._set_log_density_func()
        return self.log_density_x

    def fit(self, x=None, build_predict=True):
        """
        Trains the model from end to end. This includes preparing the model for inference,
        running the inference, and post-processing the inference results.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features), default=None
            The training instances where `n_samples` is the number of samples and `n_features`
            is the number of features.

        build_predict : bool, default=True
            Whether to build the prediction function after training.

        Returns
        -------
        self : object
            This method returns self for chaining.
        """
        self.prepare_inference(x)
        self.run_inference()
        self.process_inference(build_predict=build_predict)
        return self

    @property
    def predict(self):
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

        >>> log_density = model.predict(Xnew)

        """
        if self.log_density_func is None:
            self._set_log_density_func()
        return self.log_density_func

    def fit_predict(self, x=None, build_predict=False):
        """
        Trains the model and predicts the log density at the training points.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features), default=None
            The training instances where `n_samples` is the number of samples and `n_features`
            is the number of features.

        build_predict : bool, default=False
            Whether to build the prediction function after training.

        Raises
        ------
        ValueError
            If the input `x` is not consistent with the training data used before.

        Returns
        -------
        array-like
            The log density at each training point in `x`.
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
        return self.log_density_x
