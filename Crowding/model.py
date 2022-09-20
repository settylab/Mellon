from inspect import signature
from .conditional import build_conditional_mean, DEFAULT_SIGMA2
from .cov import Matern52
from .decomposition import build_L, DEFAULT_RANK
from .inference import inference_functions, run_inference
from .parameters import *
from .util import DEFAULT_JITTER


DEFAULT_COV_FUNC = Matern52


class CrowdingEstimator:
    R"""
    A non-parametric density estimator.
    CrowdingEstimator performs Bayesian inference, with a Gaussian process prior and Nearest
    Neighbors likelihood.

    :param mu: Mean of the Gaussian process
    :type mu: float
    :param cov_func: Gaussian process covariance function. Supports a two argument
        function or callable k(x, y) :math:`\rightarrow` float or a one argument function or class type
        that takes a length scale argument and returns a function or callable
        k(x, y) :math:`\rightarrow` float. See usage. Defaults to the type Matern52.
    :type cov_func: function or type
    :param ls: Length scale of the Gaussian process covariance function. If None,
        automatically selects the length scale based on the nearest neighbor distances.
        If cov_func is a function with two arguments, ls is ignored. Defaults to None.
    :type ls: float
    :param nn_distances: Precomputed nearest neighbor distances at each
        data point. If None, computes the nearest neighbor distances automatically, with
        a KDTree if the dimensionality of the data is less than 20, or a BallTree otherwise.
        Defaults to None.
    :type nn_distances: array-like or None
    :param initial_value: Initial guess for Maximum A Posteriori optimization. If None, finds
        :math:`z` that minimizes :math:`||Lz + mu - mle|| + ||z||`, where :math:`mle =
        \log(\text{gamma}(d/2 + 1)) - (d/2) \cdot \log(\pi) - d \cdot \log(nn\text{_}distances)`,
        where :math:`d` is the dimensionality of the data.
    :type initial_value: array-like or None
    :param L: A matrix such that :math:`L L^T \approx K`, where :math:`K` is the covariance matrix.
    :type L: array-like
    :param landmarks: Points to quantize the data for the approximate covariance
        matrix. If landmarks is an int, landmark points are selected as k-means centroids with
        k=landmarks. Defaults to the minimum between 5000 and the number of training instances.
    :type landmarks: array-like or int
    :param rank: The rank of the covariance matrix. If rank is equal to
        the number of datapoints, the covariance matrix is exact and full rank. If rank
        is equal to the number of landmark points, the standard Nystrom approximation is
        used. If rank is a float greater than 0 and less than 1, the rank is reduced
        further using the QR decomposition such that the eigenvalues of the included
        eigenvectors account for the specified percentage of the total eigenvalues.
        Defaults to 0.999.
    :type rank: int or float
    :param jitter: A small amount to add to the diagonal of the covariance
        matrix for numerical stabilitity. Defaults to 1e-6.
    :type jitter: float
    :param sigma2: White noise variance for the case the rank is reduced further
        than the number of landmark points. Ignored in other cases. Must be greater
        than 0. Defaults to 1e-6.
    :type sigma2: float
    :ivar mu: Gaussian process mean.
    :ivar cov_func: Gaussian process covariance function.
    :ivar ls: Gaussian process covariance function length scale.
    :ivar nn_distances: Nearest neighbor distances for each data point.
    :ivar landmark_points: Points to compactly summarize the data.
    :ivar n_landmarks: Number of landmark points.
    :ivar rank: Rank of approximate covariance matrix or percentage of
        eigenvalues included in approximate covariance matrix.
    :ivar jitter: A small amount added to the diagonal of the covariance matrix
        for numerical stability.
    :ivar sigma2: White noise variance for the case the rank is reduced further
        than the number of landmark points.
    :ivar initial_value: Initial guess for Maximum A Posteriori optimization.
    :ivar L: A matrix such that :math:`L L^T \approx K`, where :math:`K` is the covariance matrix.
    :ivar optimize_result: All results from the optimization.
    :ivar pre_transformation: :math:`z \sim \text{Normal}(0, I)` before
        transformation to Normal:math:`(mu, K')`, where :math:`I` is the identity matrix
        and :math:`K'` is the approximate covariance matrix.
    :ivar loss: Bayesian loss.
    :ivar log_density_x: Log density at the training points.
    :ivar log_density_func: Computes the log density at arbitrary prediction points.
    """
    def __init__(self, mu=None, cov_func=DEFAULT_COV_FUNC, ls=None, nn_distances=None,
                 initial_value=None, L=None, landmarks=DEFAULT_N_LANDMARKS,
                 rank=DEFAULT_RANK, jitter=DEFAULT_JITTER, sigma2=DEFAULT_SIGMA2):
        self.mu = mu
        self.cov_func = cov_func
        self.ls = ls
        self.nn_distances = nn_distances
        if type(landmarks) is int:
            self.landmark_points = None
            self.n_landmarks = landmarks
        else:
            self.landmark_points = landmarks
            self.n_landmarks = landmarks.shape[1]
        self.rank = rank
        self.jitter = jitter
        self.sigma2 = sigma2
        self.initial_value = initial_value
        self.L = L
        self.optimize_result = None
        self.pre_transformation = None
        self.loss = None
        self.log_density_x = None
        self.log_density_func = None

    def _set_nn(self, k=1):
        if self.nn_distances is None:
            x = self.x
            self.nn_distances = compute_nn(x)
        return self.nn_distances

    def _set_landmark_points(self):
        if (self.landmark_points is None) and (self.rank != n):
            x = self.x
            n_landmarks = self.n_landmarks
            self.landmark_points = compute_landmark_points(x, n_landmarks=n_landmarks)
        return self.landmark_points

    def _set_mu(self):
        if self.mu is None:
            nn_distances = self.nn_distances
            d = self.x.shape[1]
            self.mu = compute_mu(nn_distances, d)
        return self.mu

    def _set_ls(self):
        if self.ls is None:
            nn_distances = self.nn_distances
            self.ls = compute_ls(nn_distances)
        return self.ls

    def _set_cov_func(self):
        if len(signature(self.cov_func).parameters) != 2:
            ls = self.ls
            cov_func = self.cov_func(ls)
            self.cov_func = cov_func
        return self.cov_func

    def _set_initial_value(self):
        if self.nn_distances is None:
            nn_distances = self.nn_distances
            d = self.x.shape[1]
            mu = self.mu
            L = self.L
            self.initial_value = compute_initial_value(nn_distances, d, mu, L)
        return self.initial_value

    def _set_L(self):
        if self.L is None:
            x = self.x
            cov_func = self.cov_func
            xu = self.xu
            rank = self.rank
            jitter = self.jitter
            self.L = self.build_L(x, cov_func, xu=None, rank=DEFAULT_RANK, jitter=DEFAULT_JITTER)
        return self.L

    def _prepare_inference(self):
        nn_distances = self.nn_distances
        d = self.x.shape[1]
        mu = self.mu
        L = self.L
        self.loss_func, self.transform = inference_functions(nn_distances, d, mu, L)
        return self.loss_func, self.transform

    def _inference(self):
        function = self.loss_func
        initial_value = self.initial_value
        results = run_inference(function, initial_value)
        self.optimize_result = results
        self.pre_transformation = results.params
        self.loss = results.state.fun_val
        return self.optimize_result, self.pre_transformation, self.loss

    def _set_log_density_x(self):
        pre_transformation = self.pre_transformation
        transform = self.transform
        self.log_density_x = transform(pre_transformation)
        return self.log_density_x

    def _set_log_density_func_(self):
        rank = self.rank
        x = self.x
        xu = self.xu
        pre_transformation = self.pre_transformation
        mu = self.mu
        L = self.L
        log_density_x = self.log_density_x
        cov_func = self.cov_func
        sigma2 = self.sigma2
        log_density_func = build_conditional_mean(x, xu, rank, mu,
                                                  pre_transformation=pre_transformation,
                                                  log_density_x=log_density_x,
                                                  L=L, cov_func=cov_func, sigma2=sigma2)
        self.log_density_func = log_density_func
        return self.log_density_func

    def fit(self, x):
        R"""
        Perform Bayesian Inference.

        :param x: Training instances to estimate density function.
        :type x: array-like
        :return: self - A fitted instance of this estimator.
        :rtype: Object
        """
        self.x_ = x
        self._set_nn()
        self._set_landmark_points()
        self._set_mu()
        self._set_ls()
        self._set_cov_func()
        self._set_L()
        self._set_initial_value()
        self._set_loss_func()
        self._inference()
        self._set_log_density_x()
        self._set_log_density_func_()
        return self

    def predict(self, x):
        R"""
        Predict the log density at each point in x. Note that in the case that the rank
        is reduced below the number of landmark points, predictions at the original
        training points may differ slightly from fit_predict due to the sigma2 noise.

        :param x: New data to predict.
        :type x: array-like
        :return: log_density - The log density at each test point in x.
        :rtype: array-like
        """
        return self.log_density_func(x)

    def fit_predict(self, x):
        R"""
        Perform Bayesian Inference and return the log density at training points.

        :param x: Training instances to estimate density function.
        :type x: array-like
        :return: log_density_x - The log density at each training point in x.
        """
        self.fit(x)
        return self.log_density_x