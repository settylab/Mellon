from jax.config import config
config.update("jax_enable_x64", True)
from jax.numpy import cumsum, searchsorted, count_nonzero, eye, ones_like
from jax.numpy import dot, sqrt
from jax.numpy import sum as arraysum
from jax.numpy.linalg import eigh, cholesky, qr
from jax.scipy.linalg import solve_triangular
from .util import stabilize, DEFAULT_JITTER


DEFAULT_RANK = 0.999


def _eigendecomposition(A, rank=DEFAULT_RANK):
    R"""
    Decompose :math:`A` into its largest rank eigenvectors and eigenvalues.

    :param A: A square matrix.
    :type A: array-like
    :param rank: The rank of the decomposition, or if rank is a float greater
    than 0 and less than 1, the rank is reduced further using the QR decomposition
    such that the eigenvalues of the included eigenvectors account for the
    specified percentage of the total eigenvalues. Defaults to 0.999.
    :type rank: int or float
    :return: :math:`s, v` - The top eigenvalues and eigenvectors.
    :rtype: array-like, array-like
    """
    s, v = eigh(A)
    if rank < 1:
        # automatically choose rank to capture some percent of the eigenvalues
        target = arraysum(s) * rank
        rank = searchsorted(cumsum(s[::-1]), target)
    p = min(count_nonzero(s > 0), rank)
    s_ = s[-p:]
    v_ = v[:, -p:]
    return s_, v_


def _full_rank(x, cov_func, jitter=DEFAULT_JITTER):
    R"""
    Compute :math:`L` such that :math:`L L^T = K`, where :math:`K` is the full rank covariance matrix.

    :param x: Points.
    :type x: array-like
    :param cov_func: Covariance function.
    :type cov_func: function
    :param jitter: A small amount to add to the diagonal. Defaults to 1e-6.
    :type jitter: float
    :return: :math:`L` - A matrix such that :math:`L L^T = K`.
    :rtype: array-like
    """
    W = stabilize(cov_func(x, x), jitter)
    L = cholesky(W)
    return L


def _standard_low_rank(x, cov_func, xu, jitter=DEFAULT_JITTER):
    R"""
    Compute a low rank :math:`L` such that :math:`L L^T \approx K`, where :math:`K`
    is the full rank covariance matrix. The rank is equal to the number of
    landmark points.

    :param x: Points.
    :type x: array-like
    :param cov_func: Covariance function.
    :type cov_func: function
    :param xu: Landmark points.
    :type xu: array-like
    :param jitter: A small amount to add to the diagonal. Defaults to 1e-6.
    :type jitter: float
    :return: :math:`L` - A matrix such that :math:`L L^T \approx K`.
    :rtype: array-like
    """
    W = stabilize(cov_func(xu, xu), jitter)
    C = cov_func(x, xu)
    U = cholesky(W)
    L = solve_triangular(U, C.T, lower=True).T
    return L


def _modified_low_rank(x, cov_func, xu, rank=DEFAULT_RANK, jitter=DEFAULT_JITTER):
    R"""
    Compute a low rank :math:`L` such that :math:`L L^T ~= K`, where :math:`K` is the
    full rank covariance matrix. The rank is less than or equal to the number of
    landmark points.

    :param x: Points.
    :type x: array-like
    :param cov_func: Covariance function.
    :type cov_func: function
    :param xu: Landmark points.
    :type xu: array-like
    :param rank: The rank of the decomposition, or if rank is a float greater
        than 0 and less than 1, the rank is reduced further using the QR decomposition
        such that the eigenvalues of the included eigenvectors account for the
        specified percentage of the total eigenvalues. Defaults to 0.999.
    :type rank: int or float
    :param jitter: A small amount to add to the diagonal. Defaults to 1e-6.
    :type jitter: float
    :return: :math:`L` - A matrix such that :math:`L L^T \approx K`.
    :rtype: array-like
    """
    W = stabilize(cov_func(xu, xu), jitter)
    C = cov_func(x, xu)
    Q, R = qr(C, mode='reduced')
    s, v = _eigendecomposition(W, rank=xu.shape[0])
    T = R @ v
    S, V = _eigendecomposition(T / s @ T.T, rank=rank)
    L = Q @ V * sqrt(S)
    return L


def compute_L(x, cov_func, landmarks=None, rank=DEFAULT_RANK, jitter=DEFAULT_JITTER):
    R"""
    Compute a low rank :math:`L` such that :math:`L L^T \approx K`, where
    :math:`K` is the full rank covariance matrix. The rank is less than or
    equal to the number of landmark points.

    :param x: Points.
    :type x: array-like
    :param cov_func: Covariance function.
    :type cov_func: function
    :param landmarks: Points to summarize the data. If None, computes a full rank decompostion.
        Defaults to None.
    :type landmarks: array-like
    :param rank: The rank of the covariance matrix. If rank is equal to
        the number of datapoints, the covariance matrix is exact and full rank. If rank
        is equal to the number of landmark points, the standard Nystrom approximation is
        used. If rank is a float greater than 0 and less than 1, the rank is reduced
        further using the QR decomposition such that the eigenvalues of the included
        eigenvectors account for the specified percentage of the total eigenvalues.
        Defaults to 0.999.
    :type rank: int or float
    :param jitter: A small amount to add to the diagonal. Defaults to 1e-6.
    :type jitter: float
    :return: :math:`L` - A matrix such that :math:`L L^T \approx K`.
    :rtype: array-like
    """
    n = x.shape[0]
    n_landmarks = landmarks.shape[0]
    if (landmarks is None) or (rank == n):
        return _full_rank(x, cov_func, jitter=jitter)
    elif rank == landmarks.shape[0]:
        return _standard_low_rank(x, cov_func, landmarks, jitter=jitter)
    elif (rank > 0) and (rank <= n_landmarks):
        return _modified_low_rank(x, cov_func, landmarks, rank=rank, jitter=jitter)
    else:
        raise ValueError(f"""rank={rank} must be a float 0 < rank < 1 or
                             an int 1 <= rank <= {n_landmarks}, the number of
                             landmarks, or {n}, the number of datapoints.""")