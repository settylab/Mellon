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
DEFAULT_INFERENCE_FUNC = run_inference


class CrowdingEstimator:
    R"""
    A non-parametric density estimator.
    CrowdingEstimator performs Bayesian inference with a Gaussian process prior and Nearest
    Neighbor likelihood. All intermediate computations are cached as instance variables, so
    the user can access intermediate results and change some parameters without recomputing
    every step. See usage.

    :param cov_func_curry: Generator of the Gaussian process covariance function.
        Must be a curry that takes one length scale argument and returns a
        covariance function of the form k(x, y) :math:`\rightarrow` float.
        Defaults to the type Matern52.
    :type cov_func_curry: function or type
    :param n_landmarks: Number of landmark points. If less than 1 or greater than or
        equal to the number of training points, does not compute or use inducing points.
        Defaults to 5000.
    :type n_landmarks: int
    :param rank: The rank of the covariance matrix. If rank is equal to
        the number of datapoints, the covariance matrix is exact and full rank. If rank
        is equal to the number of landmark points, the standard Nystrom approximation is
        used. If rank is a float greater than 0 and less than 1, the rank is reduced
        further using the QR decomposition such that the eigenvalues of the included
        eigenvectors account for the specified percentage of the total eigenvalues.
        Defaults to 0.999.
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
    :param sigma2: White noise variance for the case the rank is reduced further
        than the number of landmark points. Ignored in other cases. Must be greater
        than 0. Defaults to 1e-6.
    :type sigma2: float
    :param n_iter: Number of optimization iterations. Defaults to 100.
    :type n_iter: int
    :param init_learn_rate: Initial learn rate. Defaults to 1.
    :type init_learn_rate: float
    :param inference_func: Function that performs Bayesian inference. The signature
        must be (loss_func, initial_value, n_iter=n_iter,
        init_learn_rate=init_learn_rate) :math:`\rightarrow`
        (pre_transformation, optimize_results). pre_transformation is the vector of
        parameters of the optimization at the final step and optimize_results
        contains any other results to save. Defaults to run_inference from the
        inference module.
    :type inference_func: function
    :param landmarks: Points to quantize the data for the approximate covariance. If None,
        landmarks are set as k-means centroids with k=n_landmarks. Ignored if n_landmarks
        is greater than or equal to the number of training points. Defaults to None.
    :type landmarks: array-like or None
    :param nn_distances: Precomputed nearest neighbor distances at each
        data point. If None, computes the nearest neighbor distances automatically, with
        a KDTree if the dimensionality of the data is less than 20, or a BallTree otherwise.
        Defaults to None.
    :type nn_distances: array-like or None
    :param d: Local dimensionality of the data. If None, sets d to the size of axis 1
        of the training data points. Defaults to None.
    :type d: int or None
    :param mu: Mean of the Gaussian process. If None, sets mu to the 1th percentile
        of :math:`mle(nn\text{_}distances, d) - 10`, where :math:`mle = \log(\text{gamma}(d/2 + 1))
        - (d/2) \cdot \log(\pi) - d \cdot \log(nn\text{_}distances)`. Defaults to None.
    :type mu: float or None
    :param ls: Length scale of the Gaussian process covariance function. If None,
        sets ls to the geometric mean of the nearest neighbor distances times a constant.
        If cov_func is supplied explictly, ls has no effect. Defaults to None.
    :type ls: float or None
    :param cov_func: Gaussian process covariance function of the form
        k(x, y) :math:`\rightarrow` float. If None, automatically generates the covariance
        function cov_func = cov_func_curry(ls). Defaults to None.
    :type cov_func: function or None
    :param L: A matrix such that :math:`L L^T \approx K`, where :math:`K` is the covariance matrix.
        If None, automatically computes L. Defaults to None.
    :type L: array-like or None
    :param initial_value: Initial guess for Maximum A Posteriori optimization. If None, finds
        :math:`z` that minimizes :math:`||Lz + mu - mle|| + ||z||`, where :math:`mle =
        \log(\text{gamma}(d/2 + 1)) - (d/2) \cdot \log(\pi) - d \cdot \log(nn\text{_}distances)`,
        where :math:`d` is the dimensionality of the data. Defaults to None.
    :type initial_value: array-like or None
    :ivar cov_func_curry: Generator of the Gaussian process covariance function.
    :ivar n_landmarks: Number of landmark points.
    :ivar rank: Rank of approximate covariance matrix or percentage of
        eigenvalues included in approximate covariance matrix.
    :ivar method: Method to interpret the rank as a fixed number of eigenvectors
        or a percentage of eigenvalues.
    :ivar jitter: A small amount added to the diagonal of the covariance matrix
        for numerical stability.
    :ivar sigma2: White noise variance for the case the rank is reduced further
        than the number of landmark points.
    :ivar landmarks: Points to quantize the data.
    :ivar nn_distances: Nearest neighbor distances for each data point.
    :ivar d: Local dimensionality of the data.
    :ivar mu: Gaussian process mean.
    :ivar ls: Gaussian process covariance function length scale.
    :ivar cov_func: Gaussian process covariance function.
    :ivar L: A matrix such that :math:`L L^T \approx K`, where :math:`K` is the covariance matrix.
    :ivar initial_value: Initial guess for Maximum A Posteriori optimization.
    :ivar x: The training data.
    :ivar transform: A function :math:`z \sim \text{Normal}(0, I) \rightarrow \text{Normal}(mu, K')`.
    :ivar loss_func: Bayesian loss function.
    :ivar pre_transformation: :math:`z \sim \text{Normal}(0, I)` before
        transformation to :math:`\text{Normal}(mu, K')`, where :math:`I` is the identity matrix
        and :math:`K'` is the approximate covariance matrix.
    :ivar optimize_result: All results from the optimization. By default is a tuple containing
        the last step of the optimization and the Bayesian losses at each step.
    :ivar log_density_x: Log density at the training points.
    :ivar log_density_func: Computes the log density at arbitrary prediction points.
    """
    def __init__(self, cov_func_curry=DEFAULT_COV_FUNC, \
                 n_landmarks=DEFAULT_N_LANDMARKS, \
                 rank=DEFAULT_RANK, method=DEFAULT_METHOD, \
                 jitter=DEFAULT_JITTER, sigma2=DEFAULT_SIGMA2, \
                 n_iter=DEFAULT_N_ITER, init_learn_rate=DEFAULT_INIT_LEARN_RATE, \
                 inference_func=DEFAULT_INFERENCE_FUNC, \
                 landmarks=None, nn_distances=None, d=None, mu=None, \
                 ls=None, cov_func=None, L=None, initial_value=None):
        self._implicit = set()
        self.cov_func_curry = cov_func_curry
        self.n_landmarks = n_landmarks
        self.rank = rank
        self.method = method
        self.jitter = jitter
        self.sigma2 = sigma2
        self.n_iter = n_iter
        self.init_learn_rate = init_learn_rate
        self.inference_func = inference_func
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
        self._dependencies = {'x': ['d', 'nn_distances', 'landmarks', 'L', 'log_density_func'],
                              'd': ['mu', 'initial_value', 'loss_func'],
                              'mu': ['initial_value', 'transform', 'log_density_func'],
                              'cov_func_curry': ['cov_func'],
                              'cov_func': ['L', 'log_density_func'],
                              'ls': ['cov_func'],
                              'nn_distances': ['ls', 'mu', 'initial_value', 'loss_func'],
                              'initial_value': ['pre_transformation', 'optimize_result'],
                              'n_iter': ['pre_transformation', 'optimize_result'],
                              'init_learn_rate': ['pre_transformation', 'optimize_result'],
                              'inference_func': ['pre_transformation', 'optimize_result'],
                              'n_landmarks': ['landmarks'],
                              'landmarks': ['L', 'log_density_func'],
                              'rank': ['L', 'log_density_func'],
                              'method': ['L'],
                              'jitter': ['L'],
                              'sigma2': ['log_density_func'],
                              'L': ['initial_value', 'transform', 'log_density_func'],
                              'transform': ['loss_func', 'log_density_x'],
                              'loss_func': ['pre_transformation', 'optimize_result'],
                              'optimize_result': [],
                              'pre_transformation': ['log_density_func'],
                              'log_density_x': ['log_density_func'],
                              'log_density_func': [],
                              }

    def _set_x(self, x):
        self.recursive_setattr('x', x)
        return self.x

    def _compute_landmarks(self):
        if (self.landmarks is None):
            x = self.x
            n_landmarks = self.n_landmarks
            landmarks = compute_landmarks(x, n_landmarks=n_landmarks)
            self._implicit_setattr('landmarks', landmarks)
        return self.landmarks

    def _compute_nn_distances(self):
        if self.nn_distances is None:
            x = self.x
            nn_distances = compute_nn_distances(x)
            self._implicit_setattr('nn_distances', nn_distances)
        return self.nn_distances

    def _compute_d(self):
        if self.d is None:
            d = compute_d(x)
            if d > 50:
                message = f"""Detected dimensionality of the data is over 50,
                which is likely to cause numerical instability issues.
                Consider running a dimensionality reduction algorithm, or
                if this number of dimensions is intended, explicitly pass
                d={self.d} as a parameter."""
                raise ValueError(message)
            self._implicit_setattr('d', d)
        return self.d

    def _compute_mu(self):
        if self.mu is None:
            nn_distances = self.nn_distances
            d = self.d
            mu = compute_mu(nn_distances, d)
            self._implicit_setattr('mu', mu)
        return self.mu

    def _compute_ls(self):
        if self.ls is None:
            nn_distances = self.nn_distances
            ls = compute_ls(nn_distances)
            self._implicit_setattr('ls', ls)
        return self.ls

    def _compute_cov_func(self):
        if self.cov_func is None:
            ls = self.ls
            cov_func_curry(ls)
            cov_func = compute_cov_func(cov_func_curry, ls)
            self._implicit_setattr('cov_func', cov_func)
        return self.cov_func

    def _compute_L(self):
        if self.L is None:
            x = self.x
            cov_func = self.cov_func
            landmarks = self.landmarks
            rank = self.rank
            jitter = self.jitter
            L = compute_L(x, cov_func, landmarks=landmarks, rank=rank, method=method, jitter=jitter)
            self._implicit_setattr('L', L)
        return self.L

    def _compute_initial_value(self):
        if self.initial_value is None:
            nn_distances = self.nn_distances
            d = self.d
            mu = self.mu
            L = self.L
            initial_value = compute_initial_value(nn_distances, d, mu, L)
            self._implicit_setattr('initial_value', initial_value)
        return self.initial_value

    def _compute_transform(self):
        if self.transform is None:
            mu = self.mu
            L = self.L
            transform = compute_transform(mu, L)
            self._implicit_setattr('transform', transform)
        return self.transform

    def _compute_loss_func(self):
        if self.loss_func is None:
            nn_distances = self.nn_distances
            d = self.d
            transform = self.transform
            k = self.initial_value.shape[0]
            loss_func = compute_loss_func(nn_distances, d, transform, k)
            self._implicit_setattr('loss_func', loss_func)
        return self.loss_func

    def _run_inference(self):
        if self.pre_transformation is None:
            function = self.loss_func
            initial_value = self.initial_value
            n_iter = self.n_iter
            init_learn_rate = self.init_learn_rate
            pre_transformation, optimize_results = inference_func(function, initial_value, \
                                                                 n_iter=n_iter, \
                                                                 init_learn_rate=init_learn_rate)

            self._implicit_setattr('pre_transformation', pre_transformation)
            self._implicit_setattr('optimize_results', optimize_results)
        return inference_dictionary

    def _compute_log_density_x(self):
        if self.log_density_x is None:
            pre_transformation = self.pre_transformation
            transform = self.transform
            log_density_x = compute_log_density_x(pre_transformation, transform)
            self._implicit_setattr('log_density_x', log_density_x)
        return self.log_density_x

    def _compute_log_density_func_(self):
        if self.log_density_func is None:
            rank = self.rank
            x = self.x
            landmarks = self.landmarks
            pre_transformation = self.pre_transformation
            mu = self.mu
            L = self.L
            log_density_x = self.log_density_x
            cov_func = self.cov_func
            sigma2 = self.sigma2
            log_density_func = compute_conditional_mean(rank, mu, cov_func, x=x, landmarks=landmarks, 
                                                        pre_transformation=pre_transformation,
                                                        log_density_x=log_density_x,
                                                        L=L, sigma2=sigma2)
            self._implicit_setattr('log_density_func', log_density_func)
        return self.log_density_func

    def fit(self, x):
        R"""
        Perform Bayesian Inference.

        :param x: Training instances to estimate density function.
        :type x: array-like
        :return: self - A fitted instance of this estimator.
        :rtype: Object
        """
        self._set_x(x)
        self._compute_landmarks()
        self._compute_nn()
        self._compute_d()
        self._compute_mu()
        self._compute_ls()
        self._compute_cov_func()
        self._compute_L()
        self._compute_initial_value()
        self._compute_transform()
        self._compute_loss_func()
        self._run_inference()
        self._compute_log_density_x()
        self._compute_log_density_func_()
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
        Perform Bayesian inference and return the log density at training points.

        :param x: Training instances to estimate density function.
        :type x: array-like
        :return: log_density_x - The log density at each training point in x.
        """
        self.fit(x)
        return self.log_density_x

    def recursive_setattr(self, attribute, value):
        R"""
        If value is different from the current value of attribute,
        sets attribute to value and sets any other attributes that depend
        on attribute and were computed implicitly to None. If value is 
        the current value of attribute, has no effect. Equality
        is determined by the 'is' keyword.

        :param attribute: The name of the attribute.
        :type attribute: string
        :param value: The value to set attribute to.
        :type value: anything
        """
        if self.attribute is value:
            return
        visited = set()
        self._step(attribute, visited)
        setattr(self, attribute, value)

    def _step(self, attribute, visited):
        R"""
        Recursive helper for recursive_setattr.
        """
        if attribute in visited:
            return
        else:
            visited.add(attribute)
        if attribute in self._implicit:
            setattr(self, attribute, None)
        for next_attribute in self._dependencies[attribute]:
            self._step(next_attribute, visited)

    def _implicit_setattr(self, attribute, value):
        R"""
        Sets attribute to value and adds attribute to the set of implicit attributes.

        :param attribute: The name of the attribute.
        :type attribute: string
        :param value: The value to set attribute to.
        :type value: anything
        """
        self.__dict__[attribute] = value
        self._implicit.add(attribute)

    def __setattr__(self, attribute, value):
        R"""
        Sets attribute to value and discards attribute from the set of implicit attributes.
        Overrides the special function __setattr__.

        :param attribute: The name of the attribute.
        :type attribute: string
        :param value: The value to set attribute to.
        :type value: anything
        """
        self.__dict__[attribute] = value
        self._implicit.discard(attribute)