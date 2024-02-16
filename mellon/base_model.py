import logging
from .cov import Matern52
from .inference import (
    minimize_adam,
    run_advi,
    minimize_lbfgsb,
    DEFAULT_N_ITER,
    DEFAULT_INIT_LEARN_RATE,
    DEFAULT_OPTIMIZER,
    DEFAULT_JIT,
)
from .parameters import (
    compute_rank,
    compute_n_landmarks,
    compute_landmarks,
    compute_gp_type,
    compute_nn_distances,
    compute_ls,
    compute_cov_func,
    compute_Lp,
    compute_L,
)
from .util import (
    test_rank,
    object_str,
    DEFAULT_JITTER,
    GaussianProcessType,
)
from .validation import (
    _validate_positive_int,
    _validate_positive_float,
    _validate_float_or_int,
    _validate_float,
    _validate_string,
    _validate_bool,
    _validate_array,
    _validate_float_or_iterable_numerical,
)
from .parameter_validation import (
    _validate_params,
    _validate_cov_func_curry,
    _validate_cov_func,
)


DEFAULT_COV_FUNC = Matern52
RANK_FRACTION_THRESHOLD = 0.8
SAMPLE_LANDMARK_RATIO = 10

logger = logging.getLogger("mellon")


class BaseEstimator:
    R"""
    Base class for the mellon estimators.
    """

    def __init__(
        self,
        cov_func_curry=DEFAULT_COV_FUNC,
        n_landmarks=None,
        rank=None,
        jitter=DEFAULT_JITTER,
        optimizer=DEFAULT_OPTIMIZER,
        n_iter=DEFAULT_N_ITER,
        init_learn_rate=DEFAULT_INIT_LEARN_RATE,
        landmarks=None,
        gp_type=None or GaussianProcessType,
        nn_distances=None,
        d=None,
        mu=0,
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
        self.cov_func_curry = _validate_cov_func_curry(
            cov_func_curry, cov_func, "cov_func_curry"
        )
        self.n_landmarks = _validate_positive_int(
            n_landmarks, "n_landmarks", optional=True
        )
        self.rank = _validate_float_or_int(rank, "rank", optional=True)
        self.jitter = _validate_positive_float(jitter, "jitter")
        self.landmarks = _validate_array(landmarks, "landmarks", optional=True)
        self.gp_type = GaussianProcessType.from_string(gp_type, optional=True)
        self.nn_distances = _validate_array(nn_distances, "nn_distances", optional=True)
        self.mu = _validate_float(mu, "mu", optional=True)
        self.ls = _validate_positive_float(ls, "ls", optional=True)
        self.ls_factor = _validate_positive_float(ls_factor, "ls_factor")
        self.cov_func = _validate_cov_func(cov_func, "cov_func", optional=True)
        self.Lp = _validate_array(Lp, "Lp", optional=True)
        self.L = _validate_array(L, "L", optional=True)
        self.d = _validate_float_or_iterable_numerical(
            d, "d", optional=True, positive=True
        )
        self.initial_value = _validate_array(
            initial_value, "initial_value", optional=True
        )
        self.optimizer = _validate_string(
            optimizer, "optimizer", choices={"adam", "advi", "L-BFGS-B"}
        )
        self.n_iter = _validate_positive_int(n_iter, "n_iter")
        self.init_learn_rate = _validate_positive_float(
            init_learn_rate, "init_learn_rate"
        )
        self.predictor_with_uncertainty = _validate_bool(
            predictor_with_uncertainty, "predictor_with_uncertainty"
        )
        self.jit = _validate_bool(jit, "jit")
        self.check_rank = _validate_bool(check_rank, "check_rank", optional=True)
        self.x = None
        self.pre_transformation = None

    def __str__(self):
        return self.__repr__()

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
            f"\n    cov_func_curry={self.cov_func_curry},"
            f"\n    n_landmarks={self.n_landmarks},"
            f"\n    rank={self.rank},"
            f"\n    gp_type={self.gp_type},"
            f"\n    jitter={self.jitter}, "
            f"\n    optimizer={self.optimizer},"
            f"\n    landmarks={landmarks},"
            f"\n    nn_distances={nn_distances},"
            f"\n    d={d},"
            f"\n    mu={self.mu},"
            f"\n    ls={self.ls},"
            f"\n    ls_factor={self.ls_factor},"
            f"\n    cov_func={self.cov_func},"
            f"\n    Lp={Lp},"
            f"\n    L={L},"
            f"\n    initial_value={initial_value},"
            f"\n    predictor_with_uncertainty={self.predictor_with_uncertainty},"
            f"\n    jit={self.jit},"
            f"\n    check_rank={self.check_rank},"
            "\n)"
        )
        return string

    def __call__(self, x=None):
        """This calls self.fit_predict(x):
        Fit model and make prediction on training data x.

        :param x: Data points.
        :type x: array-like
        :return: Predictions.
        :rtype: array-like
        """
        return self.fit_predict(x=x)

    def set_x(self, x):
        """
        Sets the training instances (x) for the model and validates that they are
        formatted correctly.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The training instances where `n_samples` is the number of samples and `n_features`
            is the number of features. Each sample is an array of features representing a
            point in the feature space.

        Returns
        -------
        array-like of shape (n_samples, n_features)
            The validated training instances.

        Raises
        ------
        ValueError
            If the input `x` is not valid. For instance, `x` may not be valid if it is not
            a numerical array-like object, or if its shape does not match the required shape
            (n_samples, n_features).
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
        self.x = _validate_array(x, "x")
        return self.x

    def _compute_n_landmarks(self):
        gp_type = self.gp_type
        n_samples = self.x.shape[0]
        landmarks = self.landmarks
        n_landmarks = compute_n_landmarks(gp_type, n_samples, landmarks)
        return n_landmarks

    def _compute_landmarks(self):
        x = self.x
        n_landmarks = self.n_landmarks
        n_samples = x.shape[0]
        if n_samples > 100 * n_landmarks and n_samples > 1e6:
            logger.info(
                f"Large number of {n_samples:,} cells and "
                f"small number of {n_landmarks:,} landmarks. Consider "
                "computing k-means on a subset of cells and passing "
                "the results as 'landmarks' to speed up the process."
            )
        landmarks = compute_landmarks(x, n_landmarks=n_landmarks)
        return landmarks

    def _compute_rank(self):
        gp_type = self.gp_type
        rank = compute_rank(gp_type)
        return rank

    def _compute_gp_type(self):
        n_landmarks = self.n_landmarks
        rank = self.rank
        n_samples = self.x.shape[0]
        gp_type = compute_gp_type(n_landmarks, rank, n_samples)
        return gp_type

    def _compute_nn_distances(self):
        x = self.x
        logger.info("Computing nearest neighbor distances.")
        nn_distances = compute_nn_distances(x)
        return nn_distances

    def _compute_ls(self):
        nn_distances = self.nn_distances
        ls = compute_ls(nn_distances)
        ls *= self.ls_factor
        return ls

    def _compute_cov_func(self):
        cov_func_curry = self.cov_func_curry
        ls = self.ls
        cov_func = compute_cov_func(cov_func_curry, ls)
        logger.info("Using covariance function %s.", str(cov_func))
        return cov_func

    def _compute_Lp(self):
        """
        This function calculates the lower triangular matrix L that is needed for
        computations involving the predictive of the Gaussian Process model.

        It has the side effect of settling self.L
        """
        x = self.x
        cov_func = self.cov_func
        gp_type = self.gp_type
        landmarks = self.landmarks
        jitter = self.jitter
        Lp = compute_Lp(
            x,
            cov_func,
            gp_type,
            landmarks,
            sigma=0,
            jitter=jitter,
        )
        return Lp

    def _compute_L(self):
        """
        This function calculates the lower triangular matrix L that is needed for
        computations involving the covariance matrix of the Gaussian Process model.

        It has the side effect of settling self.Lp
        """
        x = self.x
        cov_func = self.cov_func
        gp_type = self.gp_type
        landmarks = self.landmarks
        Lp = self.Lp
        rank = self.rank
        jitter = self.jitter
        check_rank = self.check_rank

        L = compute_L(
            x,
            cov_func,
            gp_type,
            landmarks=landmarks,
            Lp=Lp,
            rank=rank,
            sigma=0,
            jitter=jitter,
        )

        new_rank = L.shape[1]
        n_samples = x.shape[0]
        if landmarks is None:
            n_landmarks = n_samples
        else:
            n_landmarks = landmarks.shape[0]

        # Check if the new rank is too high in comparison to the number of landmarks
        if (
            gp_type == GaussianProcessType.SPARSE_NYSTROEM
            or gp_type == GaussianProcessType.FULL_NYSTROEM
        ) and new_rank > (rank * RANK_FRACTION_THRESHOLD * n_landmarks):
            logger.warning(
                f"Shallow rank reduction from {n_landmarks:,} to {new_rank:,} "
                "indicates underrepresentation by landmarks. Consider "
                "increasing n_landmarks!"
            )

        # Check if the number of landmarks is sufficient for the number of samples
        if (
            check_rank is None
            and gp_type == GaussianProcessType.SPARSE_CHOLESKY
            and SAMPLE_LANDMARK_RATIO * n_landmarks < n_samples
        ) or (check_rank is not None and check_rank):
            logger.info(
                "Estimating approximation accuracy "
                f"since {n_samples:,} samples are more than {SAMPLE_LANDMARK_RATIO} x "
                f"{n_landmarks:,} landmarks."
            )
            test_rank(L, threshold=RANK_FRACTION_THRESHOLD)
        logger.info(f"Using rank {new_rank:,} covariance representation.")

        return L

    def _validate_parameter(self):
        """
        Make sure there are no contradictions in the parameter settings.
        """
        rank = self.rank
        gp_type = self.gp_type
        n_samples = self.x.shape[0]
        n_landmarks = self.n_landmarks
        landmarks = self.landmarks
        _validate_params(rank, gp_type, n_samples, n_landmarks, landmarks)

    def _run_inference(self):
        function = self.loss_func
        initial_value = self.initial_value
        n_iter = self.n_iter
        init_learn_rate = self.init_learn_rate
        optimizer = self.optimizer
        logger.info("Running inference using %s.", optimizer)
        if optimizer == "adam":
            results = minimize_adam(
                function,
                initial_value,
                n_iter=n_iter,
                init_learn_rate=init_learn_rate,
                jit=self.jit,
            )
            self.pre_transformation = results.pre_transformation
            self.pre_transformation_std = None
            self.opt_state = results.opt_state
            self.losses = results.losses
        elif optimizer == "advi":
            results = run_advi(
                function,
                initial_value,
                n_iter=n_iter,
                init_learn_rate=init_learn_rate,
                jit=self.jit,
            )
            self.pre_transformation = results.pre_transformation
            self.pre_transformation_std = results.pre_transformation_std
            self.losses = results.losses
        elif optimizer == "L-BFGS-B":
            results = minimize_lbfgsb(
                function,
                initial_value,
                jit=self.jit,
            )
            self.pre_transformation = results.pre_transformation
            self.pre_transformation_std = None
            self.opt_state = results.opt_state
            self.losses = [
                results.loss,
            ]
        else:
            error = ValueError(
                f"Unknown optimizer {optimizer}. You can use .loss_func and "
                ".initial_value as loss function and initial state for an "
                "external optimization. Write optimal state to "
                ".pre_transformation to enable prediction with .predict()."
            )
            logger.error(error)
            raise error

    def _prepare_attribute(self, attribute):
        R"""
        If self.attribute is None, sets self.attribute to the value of its
        corresponding _compute_attribute function. If self.attribute is None, does nothing.

        :param attribute: The name of the attribute.
        :type attribute: str
        """
        if getattr(self, attribute) is not None:
            return
        function_name = "_compute_" + attribute
        function = getattr(self, function_name)
        value = function()
        setattr(self, attribute, value)

    def prepare_inference(self, x):
        R"""
        Set all attributes in preparation for fitting. It is not necessary
        to call this function before calling fit.
        """
        ...

    def fit(self):
        R"""
        Fit the model.
        """
        ...

    @property
    def predict(self):
        R"""
        Make prediction for new data x.

        :param x: Data points.
        :type x: array-like
        :return: Predictions.
        :rtype: array-like
        """
        ...

    def fit_predict(self, x):
        R"""
        Fit model and make prediction on training data x.

        :param x: Data points.
        :type x: array-like
        :return: Predictions.
        :rtype: array-like
        """
        ...
