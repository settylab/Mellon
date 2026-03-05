import logging
from jax import vmap
from jax.numpy import dot, square, isnan, any, eye, zeros, arange, ndim, isscalar, squeeze
from jax.numpy import sum as arraysum
from jax.numpy import diag as diagonal
from jax.numpy.linalg import cholesky, inv
from jax.scipy.linalg import solve_triangular, solve
from .util import ensure_2d, stabilize, DEFAULT_JITTER, add_variance
from .base_predictor import Predictor, ExpPredictor, PredictorTime
from .decomposition import DEFAULT_SIGMA


def _is_per_feature_sigma(sigma, y):
    """Check if sigma is per-feature (one sigma per output column).

    Returns True when sigma should be interpreted as a vector of per-feature
    noise standard deviations for a multi-output y of shape (n, p).
    """
    if sigma is None or isscalar(sigma) or ndim(sigma) == 0:
        return False
    # Explicit (1, p) shape
    if ndim(sigma) == 2 and sigma.shape[0] == 1 and ndim(y) == 2 and sigma.shape[1] == y.shape[1]:
        return True
    # 1D (p,) shape
    if ndim(sigma) == 1 and ndim(y) == 2 and sigma.shape[0] == y.shape[1]:
        if sigma.shape[0] == y.shape[0]:
            logger.warning(
                f"sigma length {sigma.shape[0]} matches both n_obs and n_features. "
                "Interpreting as per-feature. Pass sigma with shape (n, 1) for per-observation."
            )
        return True
    return False


def _normalize_per_feature_sigma(sigma):
    """Squeeze (1, p) sigma to (p,) for uniform internal handling."""
    if ndim(sigma) == 2 and sigma.shape[0] == 1:
        return squeeze(sigma, axis=0)
    return sigma


def _check_obs_variance(obj):
    if not hasattr(obj, "variance_weights"):
        raise ValueError(
            "The predictor was computed without obs_variance. "
            "Recompute setting `obs_variance=True`."
        )


logger = logging.getLogger("mellon")


def _get_L(x, cov_func, jitter=DEFAULT_JITTER, y_cov_factor=None):
    K = cov_func(x, x)
    K = add_variance(K, y_cov_factor, jitter=jitter)
    L = cholesky(K)
    if any(isnan(L)):
        message = (
            f"Covariance not positively definite with jitter={jitter}. "
            "Consider increasing the jitter for numerical stabilization."
        )
        logger.error(message)
        raise ValueError(message)
    return L


def _check_covariance(obj):
    if not hasattr(obj, "L"):
        raise ValueError(
            "The predictor was computed without covariance. "
            "Recompute setting `with_uncertainty=True.`"
        )


def _check_uncertainty(obj):
    if not hasattr(obj, "W"):
        raise ValueError(
            "The predictor was computed without uncertainty, e.g., using ADVI. "
            "Recompute setting `with_uncertainty=True.` and define `pre_transformation_std`"
            ", e.g., by using `optimizer='advi'`."
        )


def _sigma_to_y_cov_factor(sigma, y_cov_factor, n):
    if sigma is None and y_cov_factor is None:
        message = (
            "No input uncertainty specified. Make sure to set `sigma` or `pre_transformation_std`, "
            'e.g., by using `optimizer="advi", to quantify uncertainty of the prediction.'
        )
        logger.error(message)
        raise ValueError(message)
    if y_cov_factor is not None and sigma is not None and any(sigma > 0):
        raise ValueError(
            "One can specify either `sigma` or `y_cov_factor` to describe input noise, but not both."
        )

    if y_cov_factor is not None:
        return y_cov_factor

    sigma_ndim = ndim(sigma)
    if sigma_ndim == 0:
        y_cov_factor = eye(n) * sigma
    elif sigma_ndim == 1:
        y_cov_factor = diagonal(sigma)
    elif sigma_ndim > 1:
        # Extend sigma to higher dimensions, adding a leading dimension for the diagonal
        y_cov_factor = zeros((n,) + sigma.shape)

        def update_diag(i, ycf, val):
            return ycf.at[i, ...].set(val)

        y_cov_factor = vmap(update_diag, in_axes=(0, 0, 0), out_axes=0)(
            arange(n), y_cov_factor, sigma
        )
    else:
        raise ValueError(f"Unsupported `sigma` dimensions `{sigma_ndim}`.")

    return y_cov_factor


def _process_sigma(sigma, r, A, jitter=DEFAULT_JITTER):
    """
    Helper function to interpret and process sigma based on its shape and the shape of r.

    Args:
        sigma: Noise or covariance descriptor (scalar, element-wise, or matrix).
        r: Residual or target vector/matrix.
        A: Design or transformation matrix.

    Returns:
        Tuple (r_l, A_l):
            - r_l: Adjusted r vector/matrix.
            - A_l: Adjusted A matrix.

    Raises:
        NotImplementedError: If the sigma configuration is unsupported.
    """
    if isscalar(sigma) or sigma.shape == r.shape and r.ndim <= 1:
        logger.info("Sigma interpreted as element-wise standard deviation.")
        sigma2 = square(sigma)
        r_l = r / sigma2
        A_l = A / sigma2
    elif sigma.shape == r.shape and r.ndim > 1:
        logger.error("Sigma as distinct noise per output is not implemented.")
        raise NotImplementedError(
            "FunctionEstimator not implemented for multiple noises."
        )
    elif sigma.shape == (r.shape[0],) + r.shape and r.ndim > 1:
        logger.error(
            "Sigma as distinct covariance matrix per output is not implemented."
        )
        raise NotImplementedError(
            "FunctionEstimator not implemented for multiple covariance matrices."
        )
    elif sigma.shape == (r.shape[0], r.shape[0]):
        logger.info("Sigma interpreted as full covariance matrix.")
        L_s = cholesky(stabilize(sigma, jitter))
        r_l = solve_triangular(L_s.T, solve_triangular(L_s, r, lower=True))
        A_l = solve_triangular(L_s.T, solve_triangular(L_s, A, lower=True))
    else:
        raise ValueError("Unsupported sigma configuration.")

    return r_l, A_l


class _FullConditional:
    def __init__(
        self,
        x,
        y,
        mu,
        cov_func,
        L=None,
        sigma=DEFAULT_SIGMA,
        jitter=DEFAULT_JITTER,
        y_cov_factor=None,
        y_is_mean=False,
        with_uncertainty=False,
        obs_variance=False,
    ):
        R"""
        The mean function of the conditioned Gaussian process (GP).

        :param x: The training instances.
        :type x: array-like
        :param y: The function value at each point in x.
        :type y: array-like
        :param mu: The original GP mean.
        :type mu: float
        :param cov_func: The GP covariance function.
        :type cov_func: function
        :param L : A matrix such that :math:`L L^\top \approx K`, where :math:`K` is the
            covariance matrix of the GP.
        :type L : array-like or None
        :param sigma: Noise standard deviation of the data we condition on. Defaults to 0.
        :type sigma: float
        :param jitter: A small amount to add to the diagonal for stability. Defaults to 1e-6.
        :type jitter: float
        :param y_cov_factor: A matrix :math:`\Sigma_L` such that
            :math:`\Sigma_L\cdot\Sigma_L` is the covaraince of `y`.
            Only required if `with_uncertainty=True`. Defaults to None.
        :type y_cov_factor: array-like
        :param y_is_mean: Wether to consider y the GP mean or a noise measurment
            subject to `sigma` or `y_cov_factor`. Has no effect if `L` is passed.
            Defaults to False.
        :type y_is_mean: bool
        :param with_uncertainty: Wether to compute covariance functions and
            predictive uncertainty. Defaults to False.
        :type with_uncertainty: bool
        :param obs_variance: Whether to compute smoothed observation variance.
            Defaults to False.
        :type obs_variance: bool
        :return: conditional_mean - The conditioned GP mean function.
        :rtype: function
        """
        x = ensure_2d(x)
        original_sigma = sigma
        per_feature = _is_per_feature_sigma(sigma, y)

        if per_feature:
            sigma_pf = _normalize_per_feature_sigma(sigma)
            K = cov_func(x, x)
            n = x.shape[0]
            r = y - mu

            def _solve_one(sigma_g, r_g):
                L_g = cholesky(stabilize(K + sigma_g**2 * eye(n), jitter))
                return solve_triangular(L_g.T, solve_triangular(L_g, r_g, lower=True))

            weights = vmap(_solve_one, in_axes=(0, 1), out_axes=1)(sigma_pf, r)
        else:
            if L is None:
                logger.info("Recomputing covariance decomposition for predictive function.")
                if y_is_mean:
                    logger.debug("Assuming y is the mean of the GP.")
                    L = _get_L(x, cov_func, jitter)
                else:
                    logger.debug("Assuming y is not the mean of the GP.")
                    y_cov_factor = _sigma_to_y_cov_factor(sigma, y_cov_factor, x.shape[0])
                    sigma = None
                    L = _get_L(x, cov_func, jitter, y_cov_factor)
            r = y - mu
            weights = solve_triangular(L.T, solve_triangular(L, r, lower=True))

        self.cov_func = cov_func
        self.x = x
        self.weights = weights
        self.mu = mu
        self.jitter = jitter
        self.n_input_features = x.shape[1]
        self.n_obs = x.shape[0]

        self._state_variables = {"x", "weights", "mu", "jitter"}

        if obs_variance:
            self._compute_obs_variance(
                x, y, mu, cov_func, original_sigma, jitter, weights
            )

        if not with_uncertainty:
            return

        if per_feature:
            # Uncertainty not supported with per-feature sigma
            logger.warning(
                "with_uncertainty is not supported with per-feature sigma. Skipping."
            )
            return

        self.L = L
        self._state_variables.add("L")

        y_cov_factor = _sigma_to_y_cov_factor(sigma, y_cov_factor, x.shape[0])

        W = solve_triangular(L.T, solve_triangular(L, y_cov_factor, lower=True))
        self.W = W
        self._state_variables.add("W")

    def _compute_obs_variance(self, x, y, mu, cov_func, sigma, jitter, weights):
        """Compute smoothed observation variance using corrected residuals."""
        # Prediction at training points
        prediction = mu + dot(cov_func(x, x), weights)

        # Leverage at training points
        h = self._leverage(x, sigma)

        # Corrected squared residuals (HC3 estimator)
        residual = y - prediction
        if residual.ndim > h.ndim:
            h = h[..., None]
        corrected_r2 = residual**2 / (1 - h) ** 2

        # Fit second GP to corrected_r2 with noise regularization sigma.
        # The corrected r² are noisy (~chi²(1) scaled by σ²), so we smooth
        # them with the same noise level as the original GP.
        n = x.shape[0]
        K = cov_func(x, x)
        variance_mu = 0.0

        if ndim(sigma) >= 1:
            sigma_pf = _normalize_per_feature_sigma(sigma)

            def _var_solve_one(sigma_g, cr2_g):
                L_var = cholesky(stabilize(K + sigma_g**2 * eye(n), jitter))
                return solve_triangular(
                    L_var.T,
                    solve_triangular(L_var, cr2_g - variance_mu, lower=True),
                )

            variance_weights = vmap(_var_solve_one, in_axes=(0, 1), out_axes=1)(
                sigma_pf, corrected_r2
            )
        else:
            L_var = cholesky(stabilize(K + sigma**2 * eye(n), jitter))
            variance_weights = solve_triangular(
                L_var.T,
                solve_triangular(L_var, corrected_r2 - variance_mu, lower=True),
            )

        self.variance_weights = variance_weights
        self.variance_mu = variance_mu
        self._state_variables.add("variance_weights")
        self._state_variables.add("variance_mu")

    def _mean(self, Xnew):
        cov_func = self.cov_func
        x = self.x
        weights = self.weights
        mu = self.mu

        Kus = cov_func(Xnew, x)
        return mu + dot(Kus, weights)

    def _leverage(self, Xnew, sigma):
        cov_func = self.cov_func
        x = self.x
        jitter = self.jitter
        n = x.shape[0]

        # Hat matrix H = K (K + σ²I)^{-1}, leverage h = diag(H).
        # Using H = I - σ²(K + σ²I)^{-1}:
        #   h = 1 - σ² diag((K + σ²I)^{-1})
        # With Cholesky L of (K + σ²I):
        #   diag((K + σ²I)^{-1}) = diag(L^{-T} L^{-1}) = sum(L^{-1}², axis=0)
        K_train = cov_func(x, x)

        if ndim(sigma) >= 1:
            sigma = _normalize_per_feature_sigma(sigma)

            def _lev_one(sigma_g):
                L = cholesky(stabilize(K_train + sigma_g**2 * eye(n), jitter))
                Linv = solve_triangular(L, eye(n), lower=True)
                return 1 - sigma_g**2 * arraysum(square(Linv), axis=0)

            return vmap(_lev_one)(sigma).T  # (p, n) → (n, p)

        L = cholesky(stabilize(K_train + sigma**2 * eye(n), jitter))
        Linv = solve_triangular(L, eye(n), lower=True)
        return 1 - sigma**2 * arraysum(square(Linv), axis=0)

    def _obs_variance(self, Xnew):
        _check_obs_variance(self)
        cov_func = self.cov_func
        x = self.x
        Kus = cov_func(Xnew, x)
        return self.variance_mu + dot(Kus, self.variance_weights)

    def _covariance(self, Xnew, diag=True):
        _check_covariance(self)
        x = self.x
        cov_func = self.cov_func
        L = self.L

        Kus = cov_func(x, Xnew)
        A = solve_triangular(L, Kus, lower=True)
        if diag:
            Kss = cov_func.diag(Xnew)
            var = Kss - arraysum(square(A), axis=0)
            return var
        else:
            Kss = cov_func(Xnew, Xnew)
            cov = Kss - dot(A.T, A)
            return cov

    def _mean_covariance(self, Xnew, diag=True):
        _check_uncertainty(self)
        cov_func = self.cov_func
        x = self.x
        W = self.W

        Kus = cov_func(Xnew, x)
        cov_L = Kus @ W

        if diag:
            var = arraysum(cov_L * cov_L, axis=1)
            return var
        else:
            cov = cov_L @ cov_L.T
            return cov


class FullConditional(_FullConditional, Predictor):
    pass


class ExpFullConditional(_FullConditional, ExpPredictor):
    pass


class FullConditionalTime(_FullConditional, PredictorTime):
    pass


class _LandmarksConditional:
    def __init__(
        self,
        x,
        xu,
        y,
        mu,
        cov_func,
        L=None,
        Lp=None,
        sigma=DEFAULT_SIGMA,
        jitter=DEFAULT_JITTER,
        y_cov_factor=None,
        y_is_mean=False,
        with_uncertainty=False,
        obs_variance=False,
    ):
        R"""
        The mean function of the conditioned low rank gp, where rank
        is less than the number of landmark points.

        :param x: The training instances.
        :type x: array-like
        :param xu: The landmark points.
        :type xu: array-like
        :param y: The function value at each point in x.
        :type y: array-like
        :param mu: The original Gaussian process mean.
        :type mu: float
        :param cov_func: The Gaussian process covariance function.
        :type cov_func: function
        :param L : A matrix such that :math:`L L^\top \approx K`, where :math:`K` is the
            prior covariance matrix of the GP.
        :type L : array-like or None
        :param Lp : A matrix such that :math:`L_p L_p^\top \approx K_p`, where :math:`K_p` is the
            prior covariance matrix of the GP on the inducing points.
        :type Lp : array-like or None
        :param sigma: Noise standard deviation of the data we condition on. Defaults to 0.
        :type sigma: float
        :param jitter: A small amount to add to the diagonal for stability. Defaults to 1e-6.
        :type jitter: float
        :param y_cov_factor: A matrix :math:`\Sigma_L` such that
            :math:`\Sigma_L\cdot\Sigma_L` is the covaraince of `y`.
            Only required if `with_uncertainty=True`. Defaults to None.
        :type y_cov_factor: array-like
        :param y_is_mean: Wether to consider y the GP mean or a noise measurment
            subject to `sigma` or `y_cov_factor`. Has no effect if `L` is passed.
            Defaults to False.
        :type y_is_mean: bool
        :param with_uncertainty: Wether to compute predictive uncertainty and
            intermediate covariance functions. Defaults to False.
        :type with_uncertainty: bool
        :param obs_variance: Whether to compute smoothed observation variance.
            Defaults to False.
        :type obs_variance: bool
        :return: conditional_mean - The conditioned Gaussian process mean function.
        :rtype: function
        """
        x = ensure_2d(x)
        xu = ensure_2d(xu)
        Kuf = cov_func(xu, x)
        per_feature = _is_per_feature_sigma(sigma, y)

        if Lp is None:
            Lp = _get_L(xu, cov_func, jitter)

        A = solve_triangular(Lp, Kuf, lower=True)

        r = y - mu

        if per_feature:
            sigma_pf = _normalize_per_feature_sigma(sigma)

            def _solve_one(sigma_g, r_g):
                sigma2 = square(sigma_g)
                r_l = r_g / sigma2
                A_l = A / sigma2
                LBB = stabilize(dot(A_l, A.T), 1)
                L_B = cholesky(LBB)
                c = solve_triangular(L_B, dot(A, r_l), lower=True)
                return solve_triangular(Lp.T, solve_triangular(L_B.T, c))

            weights = vmap(_solve_one, in_axes=(0, 1), out_axes=1)(sigma_pf, r)
        else:
            if y_is_mean:
                r_l, A_l = r, A
            else:
                r_l, A_l = _process_sigma(sigma, r, A, jitter=jitter)
            LBB = stabilize(dot(A_l, A.T), 1)
            L_B = cholesky(LBB)

            c = solve_triangular(L_B, dot(A, r_l), lower=True)
            weights = solve_triangular(Lp.T, solve_triangular(L_B.T, c))

        self.cov_func = cov_func
        self.landmarks = xu
        self.weights = weights
        self.mu = mu
        self.jitter = jitter
        self.n_input_features = xu.shape[1]
        self.n_obs = x.shape[0]

        self._state_variables = {"landmarks", "weights", "mu", "jitter"}

        if obs_variance:
            self._compute_obs_variance(x, y, xu, mu, cov_func, sigma, jitter, weights)

        if not with_uncertainty:
            return

        if per_feature:
            logger.warning(
                "with_uncertainty is not supported with per-feature sigma. Skipping."
            )
            return

        self.L = Lp
        self._state_variables.add("L")

        Cs = dot(Lp, L_B)
        self.Cs = Cs
        self._state_variables.add("Cs")

        if not y_is_mean:
            return

        y_l = y_cov_factor
        C = solve_triangular(L_B, dot(A, y_l), lower=True)
        Z = solve_triangular(L_B.T, C)
        W = solve_triangular(Lp.T, Z)
        self.W = W
        self._state_variables.add("W")

    def _compute_obs_variance(self, x, y, xu, mu, cov_func, sigma, jitter, weights):
        """Compute smoothed observation variance using corrected residuals."""
        # Prediction at training points
        Kxu = cov_func(x, xu)
        prediction = mu + dot(Kxu, weights)

        # Leverage at training points
        h = self._leverage(x, sigma)

        # Corrected squared residuals (HC3 estimator)
        residual = y - prediction
        if residual.ndim > h.ndim:
            h = h[..., None]
        corrected_r2 = residual**2 / (1 - h) ** 2

        # Fit second GP on landmarks to corrected_r2 with noise sigma.
        # Uses the same regularization as the main GP for consistency.
        Lp_var = _get_L(xu, cov_func, jitter)
        Kuf_var = cov_func(xu, x)
        A_var = solve_triangular(Lp_var, Kuf_var, lower=True)
        variance_mu = 0.0

        if ndim(sigma) >= 1:
            sigma_pf = _normalize_per_feature_sigma(sigma)
            r_var = corrected_r2 - variance_mu

            def _var_solve_one(sigma_g, r_var_g):
                sigma2 = square(sigma_g)
                r_l = r_var_g / sigma2
                A_l = A_var / sigma2
                LBB = stabilize(dot(A_l, A_var.T), 1)
                L_B = cholesky(LBB)
                c = solve_triangular(L_B, dot(A_var, r_l), lower=True)
                return solve_triangular(Lp_var.T, solve_triangular(L_B.T, c))

            variance_weights = vmap(
                _var_solve_one, in_axes=(0, 1), out_axes=1
            )(sigma_pf, r_var)
        else:
            r_var = corrected_r2 - variance_mu
            r_l, A_l = _process_sigma(sigma, r_var, A_var, jitter=jitter)
            LBB_var = stabilize(dot(A_l, A_var.T), 1)
            L_B_var = cholesky(LBB_var)
            c_var = solve_triangular(L_B_var, dot(A_var, r_l), lower=True)
            variance_weights = solve_triangular(
                Lp_var.T, solve_triangular(L_B_var.T, c_var)
            )

        self.variance_weights = variance_weights
        self.variance_mu = variance_mu
        self._state_variables.add("variance_weights")
        self._state_variables.add("variance_mu")

    def _mean(self, Xnew):
        cov_func = self.cov_func
        xu = self.landmarks
        weights = self.weights
        mu = self.mu

        Kus = cov_func(Xnew, xu)
        return mu + dot(Kus, weights)

    def _leverage(self, Xnew, sigma):
        cov_func = self.cov_func
        xu = self.landmarks
        jitter = self.jitter

        B = cov_func(Xnew, xu)  # n_new x m
        if hasattr(self, "L") and self.L is not None:
            K_uu = self.L @ self.L.T
        else:
            K_uu = cov_func(xu, xu)

        if ndim(sigma) >= 1:
            sigma = _normalize_per_feature_sigma(sigma)

            def _lev_one(sigma_g):
                M = sigma_g**2 * K_uu + B.T @ B
                M = stabilize(M, jitter)
                BM = B @ inv(M)
                return arraysum(BM * B, axis=1)

            return vmap(_lev_one)(sigma).T  # (p, n) → (n, p)

        M = sigma**2 * K_uu + B.T @ B  # m x m
        M = stabilize(M, jitter)
        BM = B @ inv(M)  # n_new x m
        return arraysum(BM * B, axis=1)  # n_new

    def _obs_variance(self, Xnew):
        _check_obs_variance(self)
        cov_func = self.cov_func
        xu = self.landmarks
        Kus = cov_func(Xnew, xu)
        return self.variance_mu + dot(Kus, self.variance_weights)

    def _covariance(self, Xnew, diag=False):
        _check_covariance(self)
        cov_func = self.cov_func
        xu = self.landmarks
        L = self.L
        Cs = self.Cs

        Kus = cov_func(xu, Xnew)
        C = solve_triangular(Cs, Kus, lower=True)
        As = solve_triangular(L, Kus, lower=True)

        if diag:
            Kss = cov_func.diag(Xnew)
            var = Kss - arraysum(square(As), axis=0) + arraysum(square(C), axis=0)
            return var
        else:
            cov = cov_func(Xnew, Xnew) - dot(As.T, As) + dot(C.T, C)
            return cov

    def _mean_covariance(self, Xnew, diag=True):
        _check_uncertainty(self)
        cov_func = self.cov_func
        xu = self.landmarks
        W = self.W

        Kus = cov_func(Xnew, xu)
        cov_L = Kus @ W

        if diag:
            var = arraysum(cov_L * cov_L, axis=1)
            return var
        else:
            cov = cov_L @ cov_L.T
            return cov


class LandmarksConditional(_LandmarksConditional, Predictor):
    pass


class ExpLandmarksConditional(_LandmarksConditional, ExpPredictor):
    pass


class LandmarksConditionalTime(_LandmarksConditional, PredictorTime):
    pass


class _LandmarksConditionalCholesky:
    def __init__(
        self,
        xu,
        pre_transformation,
        mu,
        cov_func,
        n_obs,
        L=None,
        sigma=DEFAULT_SIGMA,
        jitter=DEFAULT_JITTER,
        y_is_mean=False,
        with_uncertainty=False,
        obs_variance=False,
        obs_x=None,
        obs_y=None,
    ):
        """
        The mean function of the conditioned low rank gp, where rank
        is the number of landmark points.

        :param xu: The landmark points.
        :type xu: array-like
        :param pre_transformation: The pre transform latent function representation.
        :type pre_transformation: array-like
        :param mu: The original Gaussian process mean.
        :type mu: float
        :param cov_func: The Gaussian process covariance function.
        :type cov_func: function
        :param n_obs: The number of observations/cells trained on. Used for normalization.
        :type n_obs: int
        :param L : A matrix such that :math:`L L^\top \approx K`, where :math:`K` is the
            covariance matrix of the Gaussian Process.
        :type L : array-like or None
        :param sigma: Standard deviation of `pre_transformation`. Defaults to 0.
        :type sigma: float
        :param jitter: A small amount to add to the diagonal for stability. Defaults to 1e-6.
        :type jitter: float
        :param y_is_mean: Wether to consider y the GP mean or a noise measurment
            subject to `sigma`. Has no effect if `L` is passed.
            Defaults to False.
        :type y_is_mean: bool
        :param with_uncertainty: Wether to compute predictive uncertainty and
            intermediate covariance functions. Defaults to False.
        :type with_uncertainty: bool
        :param obs_variance: Whether to compute smoothed observation variance.
            Defaults to False.
        :type obs_variance: bool
        :param obs_x: Training points, only needed when obs_variance=True.
        :type obs_x: array-like or None
        :param obs_y: Training values, only needed when obs_variance=True.
        :type obs_y: array-like or None
        :return: conditional_mean - The conditioned Gaussian process mean function.
        :rtype: function
        """
        xu = ensure_2d(xu)
        original_sigma = sigma
        if L is None:
            logger.info("Recomputing covariance decomposition for predictive function.")
            if y_is_mean:
                logger.debug("Assuming y is the mean of the GP.")
                L = _get_L(xu, cov_func, jitter)
            else:
                logger.debug("Assuming y is not the mean of the GP.")
                y_cov_factor = _sigma_to_y_cov_factor(sigma, None, xu.shape[0])
                sigma = None
                L = _get_L(xu, cov_func, jitter, y_cov_factor)

        weights = solve_triangular(L.T, pre_transformation)

        self.cov_func = cov_func
        self.landmarks = xu
        self.weights = weights
        self.mu = mu
        self.jitter = jitter
        self.n_input_features = xu.shape[1]
        self.n_obs = n_obs

        self._state_variables = {"landmarks", "weights", "mu", "jitter"}

        if obs_variance:
            if obs_x is None or obs_y is None:
                raise ValueError(
                    "obs_x and obs_y are required when obs_variance=True "
                    "for LandmarksConditionalCholesky."
                )
            self._compute_obs_variance(
                obs_x, obs_y, xu, mu, cov_func, original_sigma, jitter, weights
            )

        if not with_uncertainty:
            return

        self.L = L
        self._state_variables.add("L")

        try:
            Stds = diagonal(sigma)
        except ValueError:
            # sigma seems to be scalar
            Stds = eye(xu.shape[0]) * sigma

        W = solve_triangular(L.T, Stds)
        self.W = W
        self._state_variables.add("W")

    def _compute_obs_variance(self, x, y, xu, mu, cov_func, sigma, jitter, weights):
        """Compute smoothed observation variance using corrected residuals."""
        x = ensure_2d(x)
        # Prediction at training points
        Kxu = cov_func(x, xu)
        prediction = mu + dot(Kxu, weights)

        # Leverage at training points
        h = self._leverage(x, sigma)

        # Corrected squared residuals (HC3 estimator)
        residual = y - prediction
        if residual.ndim > h.ndim:
            h = h[..., None]
        corrected_r2 = residual**2 / (1 - h) ** 2

        # Fit second GP on landmarks to corrected_r2 with noise sigma.
        # Uses the same regularization as the main GP for consistency.
        Lp_var = _get_L(xu, cov_func, jitter)
        Kuf_var = cov_func(xu, x)
        A_var = solve_triangular(Lp_var, Kuf_var, lower=True)
        variance_mu = 0.0
        r_var = corrected_r2 - variance_mu
        r_l, A_l = _process_sigma(sigma, r_var, A_var, jitter=jitter)
        LBB_var = stabilize(dot(A_l, A_var.T), 1)
        L_B_var = cholesky(LBB_var)
        c_var = solve_triangular(L_B_var, dot(A_var, r_l), lower=True)
        variance_weights = solve_triangular(
            Lp_var.T, solve_triangular(L_B_var.T, c_var)
        )

        self.variance_weights = variance_weights
        self.variance_mu = variance_mu
        self._state_variables.add("variance_weights")
        self._state_variables.add("variance_mu")

    def _mean(self, Xnew):
        cov_func = self.cov_func
        xu = self.landmarks
        weights = self.weights
        mu = self.mu

        Kus = cov_func(Xnew, xu)
        return mu + dot(Kus, weights)

    def _leverage(self, Xnew, sigma):
        cov_func = self.cov_func
        xu = self.landmarks
        jitter = self.jitter

        B = cov_func(Xnew, xu)  # n_new x m
        if hasattr(self, "L") and self.L is not None:
            K_uu = self.L @ self.L.T
        else:
            K_uu = cov_func(xu, xu)
        M = sigma**2 * K_uu + B.T @ B  # m x m
        M = stabilize(M, jitter)
        BM = B @ inv(M)  # n_new x m
        return arraysum(BM * B, axis=1)  # n_new

    def _obs_variance(self, Xnew):
        _check_obs_variance(self)
        cov_func = self.cov_func
        xu = self.landmarks
        Kus = cov_func(Xnew, xu)
        return self.variance_mu + dot(Kus, self.variance_weights)

    def _covariance(self, Xnew, diag=True):
        _check_covariance(self)

        cov_func = self.cov_func
        landmarks = self.landmarks
        L = self.L

        K = cov_func(landmarks, Xnew)
        A = solve_triangular(L, K, lower=True)

        if diag:
            Kss = cov_func.diag(Xnew)
            var = Kss - arraysum(square(A), axis=0)
            return var
        else:
            cov = cov_func(Xnew, Xnew) - dot(A.T, A)
            return cov

    def _mean_covariance(self, Xnew, diag=True):
        _check_uncertainty(self)

        cov_func = self.cov_func
        xu = self.landmarks
        W = self.W

        Kus = cov_func(Xnew, xu)
        cov_L = Kus @ W

        if diag:
            var = arraysum(cov_L * cov_L, axis=1)
            return var
        else:
            cov = cov_L @ cov_L.T
            return cov


class LandmarksConditionalCholesky(_LandmarksConditionalCholesky, Predictor):
    pass


class ExpLandmarksConditionalCholesky(_LandmarksConditionalCholesky, ExpPredictor):
    pass


class LandmarksConditionalCholeskyTime(_LandmarksConditionalCholesky, PredictorTime):
    pass
