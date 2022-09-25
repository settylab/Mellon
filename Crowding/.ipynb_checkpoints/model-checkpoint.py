from inspect import signature
from .conditional import compute_conditional_mean, DEFAULT_SIGMA2
from .cov import Matern52
from .decomposition import compute_L, DEFAULT_RANK, DEFAULT_METHOD
from .inference import compute_transform, compute_loss_func, run_inference, \
                       compute_log_density_x, DEFAULT_N_ITER, DEFAULT_INIT_LEARN_RATE
from .parameters import compute_landmarks, compute_nn_distances, compute_d, compute_mu, \
                        compute_ls, compute_cov_func, compute_initial_value, \
                        DEFAULT_N_LANDMARKS
from .util import DEFAULT_JITTER


DEFAULT_COV_FUNC = Matern52


class CrowdingEstimator:
    R"""
    A non-parametric density estimator.
    CrowdingEstimator performs Bayesian inference with a Gaussian process prior and Nearest
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
    :param rank: The rank of the approximate covariance matrix. If using landmark points
        and rank is greater than n_landmarks, the actual rank is n_landmarks. If not using
        landmarks and rank is greater than the number of data points, the actual rank is
        the number of data points.
        If rank is an int, an :math:`n \times` rank matrix
        :math:`L` is computed such that :math:`L L^\top \approx K`, the exact
        :math:`n \times n` covariance matrix.
        If rank is a float 0.0 :math:`\le` rank :math:`\le` 1.0, the rank/size
        of :math:`L` is selected such that the included eigenvalues of the covariance
        between landmark points, or data points if not using landmark points, account
        for the specified percentage of the total eigenvalues. Defaults to 0.999.
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
        matrix for numerical stabilitity. Defaults to 1e-6.
    :type jitter: float
    :param sigma2: The white noise variance for the case the rank is reduced further
        than the number of landmark points. Ignored in other cases. Must be greater
        than 0. Defaults to 1e-6.
    :type sigma2: float
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
    :param d: The local dimensionality of the data. If None, sets d to the size of axis 1
        of the training data points. Defaults to None.
    :type d: int or None
    :param mu: The mean of the Gaussian process. If None, sets mu to the 1th percentile
        of :math:`mle(nn\text{_}distances, d) - 10`, where :math:`mle = \log(\text{gamma}(d/2 + 1))
        - (d/2) \cdot \log(\pi) - d \cdot \log(nn\text{_}distances)`. Defaults to None.
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
        :math:`z` that minimizes :math:`||Lz + mu - mle|| + ||z||`, where :math:`mle =
        \log(\text{gamma}(d/2 + 1)) - (d/2) \cdot \log(\pi) - d \cdot \log(nn\text{_}distances)`,
        where :math:`d` is the dimensionality of the data. Defaults to None.
    :type initial_value: array-like or None
    :ivar cov_func_curry: The generator of the Gaussian process covariance function.
    :ivar n_landmarks: The number of landmark points.
    :ivar rank: The rank of approximate covariance matrix or percentage of
        eigenvalues included in approximate covariance matrix.
    :ivar method: The method to interpret the rank as a fixed number of eigenvectors
        or a percentage of eigenvalues.
    :ivar jitter: A small amount added to the diagonal of the covariance matrix
        for numerical stability.
    :ivar sigma2: White noise variance for the case the rank is reduced further
        than the number of landmark points.
    :ivar n_iter: The number of optimization iterations.
    :ivar init_learn_rate: The initial learn rate.
    :ivar landmarks: The points to quantize the data.
    :ivar nn_distances: The nearest neighbor distances for each data point.
    :ivar d: The local dimensionality of the data.
    :ivar mu: The Gaussian process mean.
    :ivar ls: The Gaussian process covariance function length scale.
    :ivar cov_func: The Gaussian process covariance function.
    :ivar L: A matrix such that :math:`L L^\top \approx K`, where :math:`K` is the covariance matrix.
    :ivar initial_value: Initial guess for Maximum A Posteriori optimization.
    :ivar x: The training data.
    :ivar transform: A function :math:`z \sim \text{Normal}(0, I) \rightarrow \text{Normal}(mu, K')`.
    :ivar loss_func: The Bayesian loss function.
    :ivar pre_transformation: The optimized parameters :math:`z \sim \text{Normal}(0, I)` before
        transformation to :math:`\text{Normal}(mu, K')`, where :math:`I` is the identity matrix
        and :math:`K'` is the approximate covariance matrix.
    :ivar optimize_result: All results from the optimization. By default is a tuple containing
        the last step of the optimization and the Bayesian losses at each step.
    :ivar log_density_x: The log density at the training points.
    :ivar log_density_func: A function that computes the log density at arbitrary prediction points.
    """
    def __init__(self, cov_func_curry=DEFAULT_COV_FUNC, \
                 n_landmarks=DEFAULT_N_LANDMARKS, \
                 rank=DEFAULT_RANK, method=DEFAULT_METHOD, \
                 jitter=DEFAULT_JITTER, sigma2=DEFAULT_SIGMA2, \
                 n_iter=DEFAULT_N_ITER, init_learn_rate=DEFAULT_INIT_LEARN_RATE, \
                 landmarks=None, nn_distances=None, d=None, mu=None, \
                 ls=None, cov_func=None, L=None, initial_value=None):
        self.cov_func_curry = cov_func_curry
        self.n_landmarks = n_landmarks
        self.rank = rank
        self.method = method
        self.jitter = jitter
        self.sigma2 = sigma2
        self.n_iter = n_iter
        self.init_learn_rate = init_learn_rate
        self.landmarks = landmarks
        self.nn_distances = nn_distances
        self.d = d
        self.mu = mu
        self.ls = ls
        self.cov_func = cov_func
        self.L = L
        self.initial_value = initial_value
        self.x = None
        self.transform = None
        self.loss_func = None
        self.optimize_result = None
        self.pre_transformation = None
        self.losses = None
        self.log_density_x = None
        self.log_density_func = None

    def _set_x(self, x):
        self.x = x

    def _compute_landmarks(self):
        x = self.x
        n_landmarks = self.n_landmarks
        landmarks = compute_landmarks(x, n_landmarks=n_landmarks)
        return landmarks

    def _compute_nn_distances(self):
        x = self.x
        nn_distances = compute_nn_distances(x)
        return nn_distances

    def _compute_d(self):
        x = self.x
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

    def _compute_ls(self):
        nn_distances = self.nn_distances
        ls = compute_ls(nn_distances)
        return ls

    def _compute_cov_func(self):
        cov_func_curry = self.cov_func_curry
        ls = self.ls
        cov_func = compute_cov_func(cov_func_curry, ls)
        return cov_func

    def _compute_L(self):
        x = self.x
        cov_func = self.cov_func
        landmarks = self.landmarks
        rank = self.rank
        method = self.method
        jitter = self.jitter
        L = compute_L(x, cov_func, landmarks=landmarks, rank=rank, method=method, jitter=jitter)
        return L

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

    def _run_inference(self):
        function = self.loss_func
        initial_value = self.initial_value
        n_iter = self.n_iter
        init_learn_rate = self.init_learn_rate
        optimize_results, pre_transformation, losses = run_inference(function, initial_value, \
                                                              n_iter=n_iter, \
                                                              init_learn_rate=init_learn_rate)
        self.optimize_results = optimize_results
        self.pre_transformation = pre_transformation
        self.losses = losses

    def _set_log_density_x(self):
        pre_transformation = self.pre_transformation
        transform = self.transform
        log_density_x = compute_log_density_x(pre_transformation, transform)
        self.log_density_x = log_density_x

    def _set_log_density_func(self):
        rank = self.rank
        x = self.x
        landmarks = self.landmarks
        pre_transformation = self.pre_transformation
        mu = self.mu
        L = self.L
        log_density_x = self.log_density_x
        cov_func = self.cov_func
        sigma2 = self.sigma2
        log_density_func = compute_conditional_mean(rank, mu, cov_func, x=x, landmarks=landmarks, \
                                                    pre_transformation=pre_transformation, \
                                                    log_density_x=log_density_x, \
                                                    L=L, sigma2=sigma2)
        self.log_density_func = log_density_func

    def _prepare_attribute(self, attribute):
        R"""
        If self.attribute is None, sets self.attribute to the value of its
        corresponding _compute_attribute function. If self.attribute is None, does nothing.

        :param attribute: The name of the attribute.
        :type attribute: str
        """
        if self.__dict__[attribute] is not None:
            return
        function_name = '_compute_' + attribute
        function = CrowdingEstimator.__dict__[function_name]
        value = function(self)
        setattr(self, attribute, value)

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
        self._prepare_attribute('nn_distances')
        self._prepare_attribute('d')
        self._prepare_attribute('mu')
        self._prepare_attribute('ls')
        self._prepare_attribute('cov_func')
        self._prepare_attribute('L')
        self._prepare_attribute('initial_value')
        self._prepare_attribute('transform')
        self._prepare_attribute('loss_func')
        return self.loss_func, self.initial_value

    def run_inference(self, loss_func=None, initial_value=None):
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

    def fit(self, x, build_predict=True):
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
        if self.x is not None and self.x != x:
            message = "self.x has been set already, but is not equal to the argument x."
            raise ValueError(message)

        self.prepare_inference(x)
        self.run_inference()
        self.process_inference(build_predict=build_predict)
        return self

    def predict(self, x):
        R"""
        Predict the log density at each point in x. Note that in the case that the rank
        is reduced below the number of landmark points, predictions at the original
        training points may differ slightly from fit_predict due to the sigma2 noise.

        :param x: The new data to predict.
        :type x: array-like
        :return: log_density - The log density at each test point in x.
        :rtype: array-like
        """
        return self.log_density_func(x)

    def fit_predict(self, x):
        R"""
        Perform Bayesian inference and return the log density at training points.

        :param x: The training instances to estimate density function.
        :type x: array-like
        :return: log_density_x - The log density at each training point in x.
        """
        self.fit(x, build_predict=True)
        return self.log_density_x