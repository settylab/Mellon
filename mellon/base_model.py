from .cov import Matern52
from .decomposition import DEFAULT_RANK, DEFAULT_METHOD
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
    compute_landmarks,
    compute_nn_distances,
    compute_ls,
    compute_cov_func,
    compute_L,
    DEFAULT_N_LANDMARKS,
)
from .util import (
    test_rank,
    DEFAULT_JITTER,
    Log,
)
from .validation import (
    _validate_positive_int,
    _validate_positive_float,
    _validate_float_or_int,
    _validate_float,
    _validate_string,
    _validate_bool,
    _validate_array,
    _validate_cov_func_curry,
    _validate_cov_func,
    _validate_float_or_iterable_numerical,
)


DEFAULT_COV_FUNC = Matern52
RANK_FRACTION_THRESHOLD = 0.8
SAMPLE_LANDMARK_RATIO = 10

logger = Log()


class BaseEstimator:
    R"""
    Base class for the mellon estimators.
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
        nn_distances=None,
        d=None,
        mu=0,
        ls=None,
        ls_factor=1,
        cov_func=None,
        L=None,
        initial_value=None,
        jit=DEFAULT_JIT,
    ):
        self.cov_func_curry = _validate_cov_func_curry(
            cov_func_curry, cov_func, "cov_func_curry"
        )
        self.n_landmarks = _validate_positive_int(n_landmarks, "n_landmarks")
        self.rank = _validate_float_or_int(rank, "rank")
        self.method = _validate_string(
            method, "method", choices={"percent", "fixed", "auto"}
        )
        self.jitter = _validate_positive_float(jitter, "jitter")
        self.landmarks = _validate_array(landmarks, "landmarks", optional=True)
        self.nn_distances = _validate_array(nn_distances, "nn_distances", optional=True)
        self.mu = _validate_float(mu, "mu", optional=True)
        self.ls = _validate_positive_float(ls, "ls", optional=True)
        self.ls_factor = _validate_positive_float(ls_factor, "ls_factor")
        self.cov_func = _validate_cov_func(cov_func, "cov_func", optional=True)
        self.L = _validate_array(L, "L", optional=True)
        self.d = _validate_float_or_iterable_numerical(d, "d", optional=True)
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
        self.jit = _validate_bool(jit, "jit")
        self.x = None
        self.pre_transformation = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        name = self.__class__.__name__
        string = (
            f"{name}("
            f"cov_func_curry={self.cov_func_curry}, "
            f"n_landmarks={self.n_landmarks}, "
            f"rank={self.rank}, "
            f"jitter={self.jitter}, "
            f"landmarks={self.landmarks}, "
        )
        if self.nn_distances is None:
            string += "nn_distances=None, "
        else:
            string += "nn_distances=nn_distances, "
        string += f"mu={self.mu}, " f"ls={self.mu}, " f"cov_func={self.cov_func}, "
        if self.L is None:
            string += "L=None, "
        else:
            string += "L=L, "
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
        self.x = _validate_array(x, "x")
        return self.x

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

    def _compute_L(self):
        """
        This function calculates the lower triangular matrix L that is needed for
        computations involving the covariance matrix of the Gaussian Process model.
        """

        # Extract instance attributes
        x = self.x
        cov_func = self.cov_func
        landmarks = self.landmarks
        n_samples = x.shape[0]
        n_landmarks = n_samples if landmarks is None else landmarks.shape[0]
        rank = self.rank
        method = self.method
        jitter = self.jitter

        is_rank_full = (
            isinstance(rank, int)
            and rank == n_landmarks
            or isinstance(rank, float)
            and rank == 1.0
        )

        # Log the method and rank used for computation
        if not is_rank_full and method != "fixed":
            logger.info(
                f'Computing rank reduction using "{method}" method '
                f"retaining > {rank:.2%} of variance."
            )
        elif not is_rank_full:
            logger.info(
                f'Computing rank reduction to rank {rank} using "{method}" method.'
            )

        try:
            # Compute the lower triangular matrix L
            L = compute_L(
                x,
                cov_func,
                landmarks=landmarks,
                rank=rank,
                method=method,
                jitter=jitter,
            )
        except Exception as e:
            logger.error(f"Error during computation of L: {e}")
            raise

        new_rank = L.shape[1]

        # Check if the new rank is too high in comparison to the number of landmarks
        if (
            not is_rank_full
            and method != "fixed"
            and new_rank > (rank * RANK_FRACTION_THRESHOLD * n_landmarks)
        ):
            logger.warning(
                f"Shallow rank reduction from {n_landmarks:,} to {new_rank:,} "
                "indicates underrepresentation by landmarks. Consider "
                "increasing n_landmarks!"
            )

        # Check if the number of landmarks is sufficient for the number of samples
        if (
            is_rank_full
            and n_landmarks is not None
            and SAMPLE_LANDMARK_RATIO * n_landmarks < n_samples
        ):
            logger.info(
                "Estimating approximation accuracy "
                f"since {n_samples:,} samples are more than {SAMPLE_LANDMARK_RATIO} x "
                f"{n_landmarks:,} landmarks."
            )
            test_rank(L, threshold=RANK_FRACTION_THRESHOLD)
        logger.info(f"Using rank {new_rank:,} covariance representation.")
        return L

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
