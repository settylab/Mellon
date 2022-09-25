import warnings
from jax.numpy import cumsum, searchsorted, count_nonzero, eye, ones_like
from jax.numpy import dot, sqrt
from jax.numpy import sum as arraysum
from jax.numpy.linalg import eigh, cholesky, qr
from jax.scipy.linalg import solve_triangular
from .util import stabilize, DEFAULT_JITTER


DEFAULT_RANK = 0.999
DEFAULT_METHOD = 'auto'


def _select_method(rank, full):
    R"""
    Checks if rank is a float 0.0 :math:`\le` rank :math:`\le` 1.0 or an int
    1 0.0 :math:`\le` rank :math:`\le` 1.0 full. Returns True in the first case.
    Raises an error otherwise.

    :param rank: The rank of the decomposition, or if rank is a float greater
    than 0 and less than 1, the rank is reduced further using the QR decomposition
    such that the eigenvalues of the included eigenvectors account for the
    specified percentage of the total eigenvalues. Defaults to 0.999.
    :type rank: int or float
    :param full: The size of the exact matrix.
    :type full: int
    """
    percent = (type(rank) is float) and (0 < rank) and (rank <= 1)
    fixed = (type(rank) is int) and (1 <= rank) and (rank <= full)
    if not (percent or fixed):
        message = """rank must be a float 0.0 < rank <= 1.0 or
            an int 1 <= rank <= q. q equals the number of landmarks
            or the number of data points if there are no landmarks."""
        raise ValueError(message)
    if rank == 1:  # true if rank is 1.0 or 1
        if percent:
            message = """rank is 1.0, which is ambiguous. Because
                rank is a float, it is interpreted as the percentage of
                eigenvalues to include in the low rank approximation.
                To bypass this warning, explictly set method='percent'.
                If this is not the intended behavior, explicitly set
                method='fixed'."""
        else:
            message = """rank is 1, which is ambiguous. Because
                rank is an int, it is interpreted as the number of
                eigenvectors to include in the low rank approximation.
                To bypass this warning, explictly set method='fixed'.
                If this is not the intended behavior, explicitly set
                method='percent'."""
        raise warnings.warn(message, UserWarning)
    if percent:
        return 'percent'
    else:
        return 'fixed'


def _eigendecomposition(A, rank=DEFAULT_RANK, method=DEFAULT_METHOD):
    R"""
    Decompose :math:`A` into its largest rank eigenvectors and eigenvalues.

    :param A: A square matrix.
    :type A: array-like
    :param rank: The rank of the decomposition, or if rank is a float
        0.0 :math:`\le` rank :math:`\le` 1.0, the rank is reduced further using the QR
        decomposition such that the eigenvalues of the included eigenvectors account for
        the specified percentage of the total eigenvalues. Defaults to 0.999.
    :type rank: int or float
    :param method: Explicitly specifies whether rank is to be interpreted as a
        fixed number of eigenvectors or a percent of eigenvalues to include
        in the low rank approximation. Supports 'fixed', 'percent', or 'auto'.
        If 'auto', interprets rank as a fixed number of eigenvectors if it is
        an int and interprets rank as a percent of eigenvalues if it is a float.
        Defaults to 'auto'.
    :type method: str
    :return: :math:`s, v` - The top eigenvalues and eigenvectors.
    :rtype: array-like, array-like
    """

    if method == 'auto':
        full = A.shape[0]
        method = _select_method(rank, full)
    s, v = eigh(A)
    if method == 'percent':
        # automatically choose rank to capture some percent of the eigenvalues
        target = arraysum(s) * rank
        rank = searchsorted(cumsum(s[::-1]), target)
    p = min(count_nonzero(s > 0), rank)
    s_ = s[-p:]
    v_ = v[:, -p:]
    return s_, v_


def _full_rank(x, cov_func, jitter=DEFAULT_JITTER):
    R"""
    Compute :math:`L` such that :math:`L L^\top = K`, where :math:`K`
    is the full rank covariance matrix.

    :param x: The training instances.
    :type x: array-like
    :param cov_func: The Gaussian process covariance function.
    :type cov_func: function
    :param jitter: A small amount to add to the diagonal. Defaults to 1e-6.
    :type jitter: float
    :return: :math:`L` - A matrix such that :math:`L L^\top = K`.
    :rtype: array-like
    """
    W = stabilize(cov_func(x, x), jitter)
    L = cholesky(W)
    return L


def _full_decomposition_low_rank(x, cov_func, rank=DEFAULT_RANK,
                                 method=DEFAULT_METHOD, jitter=DEFAULT_JITTER):
    R"""
    Compute a low rank :math:`L` such that :math:`L L^\top ~= K`, where :math:`K` is the
    full rank covariance matrix. The rank is less than or equal to the number of
    landmark points.

    :param x: The training instances.
    :type x: array-like
    :param cov_func: The Gaussian process covariance function.
    :type cov_func: function
    :param rank: The rank of the decomposition, or if rank is a float greater
        than 0 and less than 1, the eigenvalues of the included eigenvectors
        account for the specified percentage of the total eigenvalues.
        Defaults to 0.999.
    :type rank: int or float
    :param jitter: A small amount to add to the diagonal. Defaults to 1e-6.
    :type jitter: float
    :param method: Explicitly specifies whether rank is to be interpreted as a
        fixed number of eigenvectors or a percent of eigenvalues to include
        in the low rank approximation. Supports 'fixed', 'percent', or 'auto'.
        If 'auto', interprets rank as a fixed number of eigenvectors if it is
        an int and interprets rank as a percent of eigenvalues if it is a float.
        Defaults to 'auto'.
    :type method: str
    :return: :math:`L` - A matrix such that :math:`L L^\top \approx K`.
    :rtype: array-like
    """
    W = cov_func(x, x)
    s, v = _eigendecomposition(W, rank=DEFAULT_RANK, method=method)
    L = v * sqrt(s)
    return L


def _standard_low_rank(x, cov_func, xu, jitter=DEFAULT_JITTER):
    R"""
    Compute a low rank :math:`L` such that :math:`L L^\top \approx K`, where :math:`K`
    is the full rank covariance matrix. The rank is equal to the number of
    landmark points.

    :param x: The training instances.
    :type x: array-like
    :param cov_func: The Gaussian process covariance function.
    :type cov_func: function
    :param xu: The landmark points.
    :type xu: array-like
    :param jitter: A small amount to add to the diagonal. Defaults to 1e-6.
    :type jitter: float
    :return: :math:`L` - A matrix such that :math:`L L^\top \approx K`.
    :rtype: array-like
    """
    W = stabilize(cov_func(xu, xu), jitter)
    C = cov_func(x, xu)
    U = cholesky(W)
    L = solve_triangular(U, C.T, lower=True).T
    return L


def _modified_low_rank(x, cov_func, xu, rank=DEFAULT_RANK,
                       method=DEFAULT_METHOD, jitter=DEFAULT_JITTER):
    R"""
    Compute a low rank :math:`L` such that :math:`L L^\top ~= K`, where :math:`K` is the
    full rank covariance matrix. The rank is less than or equal to the number of
    landmark points.

    :param x: The training instances.
    :type x: array-like
    :param cov_func: The Gaussian process covariance function.
    :type cov_func: function
    :param xu: The landmark points.
    :type xu: array-like
    :param rank: The rank of the decomposition, or if rank is a float
        0.0 :math:`\le` rank :math:`\le` 1.0, the rank is reduced further using
        the QR decomposition such that the eigenvalues of the included eigenvectors
        account for the specified percentage of the total eigenvalues. Defaults to 0.999.
    :type rank: int or float
    :param jitter: A small amount to add to the diagonal. Defaults to 1e-6.
    :type jitter: float
    :param method: Explicitly specifies whether rank is to be interpreted as a
        fixed number of eigenvectors or a percent of eigenvalues to include
        in the low rank approximation. Supports 'fixed', 'percent', or 'auto'.
        If 'auto', interprets rank as a fixed number of eigenvectors if it is
        an int and interprets rank as a percent of eigenvalues if it is a float.
        Defaults to 'auto'.
    :type method: str
    :return: :math:`L` - A matrix such that :math:`L L^\top \approx K`.
    :rtype: array-like
    """
    W = stabilize(cov_func(xu, xu), jitter)
    C = cov_func(x, xu)
    Q, R = qr(C, mode='reduced')
    s, v = _eigendecomposition(W, rank=xu.shape[0], method='fixed')
    T = R @ v
    S, V = _eigendecomposition(T / s @ T.T, rank=rank, method=method)
    L = Q @ V * sqrt(S)
    return L


def compute_L(x, cov_func, landmarks=None, rank=DEFAULT_RANK,
              method=DEFAULT_METHOD, jitter=DEFAULT_JITTER):
    R"""
    Compute an :math:`L` such that :math:`L L^\top \approx K`, where
    :math:`K` is the covariance matrix.

    :param x: The training instances.
    :type x: array-like
    :param cov_func: The Gaussian process covariance function.
    :type cov_func: function
    :param landmarks: The landmark points. If None, computes a full rank decompostion.
        Defaults to None.
    :type landmarks: array-like
    :param rank: The rank of the covariance matrix. If rank is equal to
        the number of datapoints, the covariance matrix is exact and full rank. If rank
        is equal to the number of landmark points, the standard Nystrom approximation is
        used. If rank is a float 0.0 :math:`\le` rank :math:`\le` 1.0, the rank is reduced
        further using the QR decomposition such that the eigenvalues of the included
        eigenvectors account for the specified percentage of the total eigenvalues.
        Defaults to 0.999.
    :type rank: int or float
    :param method: Explicitly specifies whether rank is to be interpreted as a
        fixed number of eigenvectors or a percent of eigenvalues to include
        in the low rank approximation. Supports 'fixed', 'percent', or 'auto'.
        If 'auto', interprets rank as a fixed number of eigenvectors if it is
        an int and interprets rank as a percent of eigenvalues if it is a float.
        Defaults to 'auto'.
    :type method: str
    :param jitter: A small amount to add to the diagonal. Defaults to 1e-6.
    :type jitter: float
    :return: :math:`L` - A matrix such that :math:`L L^\top \approx K`.
    :rtype: array-like
    """
    if landmarks is None:
        n = x.shape[0]
        if type(rank) is int and rank == n:
            return _full_rank(x, cov_func, jitter=jitter)
        else:
            return _full_decomposition_low_rank(x, cov_func, rank=rank, method=method, jitter=jitter)
    else:
        n_landmarks = landmarks.shape[0]
        if type(rank) is int and rank == n_landmarks:
            return _standard_low_rank(x, cov_func, landmarks, jitter=jitter)
        else:
            return _modified_low_rank(x, cov_func, landmarks, rank=rank, method=method, jitter=jitter)