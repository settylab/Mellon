from jax.numpy import cumsum, searchsorted, count_nonzero, sqrt, isnan, any
from jax.numpy.linalg import eigh, cholesky, qr
from jax.scipy.linalg import solve_triangular
from .util import stabilize, DEFAULT_JITTER, Log


DEFAULT_RANK = 1.0
DEFAULT_METHOD = "auto"

logger = Log()


def _check_method(rank, full, method):
    R"""
    Checks if rank is a float 0.0 :math:`\le` rank :math:`\le` 1.0 or an int
    1 :math:`\le` rank :math:`\le` full. Raises an error if neither is true
    or if method doesn't match the detected method.

    :param rank: The rank of the decomposition, or if rank is a float greater
    than 0 and less than 1, the rank is reduced further using the QR decomposition
    such that the eigenvalues of the included eigenvectors account for the
    specified percentage of the total eigenvalues. Defaults to 0.999.
    :type rank: int or float
    :param full: The size of the exact matrix.
    :type full: int
    :param method: The method to interpret the rank.
    :type method: str
    :return: method - The detected method.
    :rtype: str
    """

    percent = isinstance(rank, float) and (0 < rank) and (rank <= 1)
    fixed = isinstance(rank, int) and (1 <= rank) and (rank <= full)
    if not (percent or fixed):
        message = """rank must be a float 0.0 <=rank <= 1.0 or
    an int 1 <= rank <= q. q equals the number of landmarks
    or the number of data points if there are no landmarks."""
        raise ValueError(message)
    elif percent and not (method == "percent" or method == "auto"):
        message = f"""The argument method={method} does not match the rank={rank}.
    The detected method from the rank is 'percent'."""
        raise ValueError(message)
    elif fixed and not (method == "fixed" or method == "auto"):
        message = f"""The argument method={method} does not match the rank={rank}.
    The detected method from the rank is 'fixed'."""
        raise ValueError(message)
    if percent:
        return "percent"
    else:
        return "fixed"


def _eigendecomposition(A, rank=DEFAULT_RANK, method=DEFAULT_METHOD):
    R"""
    Decompose :math:`A` into its largest positive `rank` and
    at least one eigenvector(s) and eigenvalue(s).

    :param A: A square matrix.
    :type A: array-like
    :param rank: The rank of the decomposition, or if rank is a float
        0.0 :math:`\le` rank :math:`\le` 1.0, the rank is reduced further using the QR
        decomposition such that the eigenvalues of the included eigenvectors account for
        the specified percentage of the total eigenvalues. Defaults to 0.999.
    :type rank: int or float
    :param method: Explicitly specifies whether rank is to be interpreted as a
        fixed number of eigenvectors or a percent of eigenvalues to include
        in the low rank approximation.
    :type method: str
    :return: :math:`s, v` - The top eigenvalues and eigenvectors.
    :rtype: array-like, array-like
    """

    s, v = eigh(A)
    if any(s <= 0):
        message = (
            "Singuarity detected in covariance matrix. "
            "This can complicated prediction. Consider raising the jitter."
        )
        logger.warning(message)
    p = count_nonzero(s > 0)  # stability
    summed = cumsum(s[: -p - 1 : -1])
    if method == "percent":
        # automatically choose rank to capture some percent of the eigenvalues
        target = summed[-1] * rank
        p = searchsorted(summed, target)
        if p == 0:
            logger.warning(
                f"Low variance percentage {rank:%} indicated rank=0. "
                "Bumping rank to 1."
            )
            p = 1
    else:
        p = min(rank, p)
    if (method == "percent" and rank < 1) or rank < len(summed):
        frac = summed[p] / summed[-1]
        logger.info(f"Recovering {frac:%} variance in eigendecomposition.")
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
    if any(isnan(L)):
        message = (
            f"Covariance not positively definite with jitter={jitter}. "
            "Consider increasing the jitter for numerical stabilization."
        )
        logger.error(message)
        raise ValueError(message)
    return L


def _full_decomposition_low_rank(
    x, cov_func, rank=DEFAULT_RANK, method=DEFAULT_METHOD, jitter=DEFAULT_JITTER
):
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
    s, v = _eigendecomposition(W, rank=rank, method=method)
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
    if any(isnan(U)):
        message = (
            f"Covariance of landmarks not positively definite with jitter={jitter}. "
            "Consider increasing the jitter for numerical stabilization."
        )
        logger.error(message)
        raise ValueError(message)
    L = solve_triangular(U, C.T, lower=True).T
    return L


def _modified_low_rank(
    x, cov_func, xu, rank=DEFAULT_RANK, method=DEFAULT_METHOD, jitter=DEFAULT_JITTER
):
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
    Q, R = qr(C, mode="reduced")
    s, v = _eigendecomposition(W, rank=xu.shape[0], method="fixed")
    T = R @ v
    S, V = _eigendecomposition(T / s @ T.T, rank=rank, method=method)
    L = Q @ V * sqrt(S)
    return L
