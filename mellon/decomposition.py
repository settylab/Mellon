import logging
from jax.numpy import (
    cumsum,
    searchsorted,
    count_nonzero,
    sqrt,
    isnan,
    any,
    where,
    square,
)
from jax.numpy.linalg import eigh, cholesky, qr
from jax.scipy.linalg import solve_triangular
from .util import stabilize, DEFAULT_JITTER


DEFAULT_RANK = 0.99
DEFAULT_SIGMA = 0

logger = logging.getLogger("mellon")


def _eigendecomposition(A, rank=DEFAULT_RANK):
    R"""
    Decompose :math:`A` into its largest positive `rank` and
    at least one eigenvector(s) and eigenvalue(s).

    Parameters
    ----------
    A : array-like
        A square matrix.
    rank : int or float, optional
        The rank of the decomposition, or if rank is a float
        0.0 :math:`\le` rank :math:`\le` 1.0, the rank is reduced further using the QR
        decomposition such that the eigenvalues of the included eigenvectors account for
        the specified percentage of the total eigenvalues. Defaults to 0.99.

    Returns
    -------
    array-like, array-like
        :math:`s, v` - The top eigenvalues and eigenvectors.

    Notes
    -----
    If any eigenvalues are less than or equal to 0, a warning message will be logged,
    indicating a singularity in the covariance matrix. Consider raising the jitter
    value to address this issue.
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
    if isinstance(rank, float):
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
    if (isinstance(rank, float) and rank < 1) or rank < len(summed):
        frac = summed[p] / summed[-1]
        logger.info(f"Recovering {frac:%} variance in eigendecomposition.")
    s_ = s[-p:]
    v_ = v[:, -p:]
    return s_, v_


def _full_rank(x, cov_func, sigma=DEFAULT_SIGMA, jitter=DEFAULT_JITTER):
    R"""
    Compute :math:`L` such that :math:`L L^\top = K`, where :math:`K`
    is the full rank covariance matrix.

    Parameters
    ----------
    x : array-like
        The training instances.
    cov_func : function
        The Gaussian process covariance function.
    sigma : float, optional
        Noise standard deviation of the data we condition on. Defaults to 0.
    jitter : float, optional
        A small amount to add to the diagonal. Defaults to 1e-6.

    Returns
    -------
    array-like
        :math:`L` - A matrix such that :math:`L L^\top = K`.

    Raises
    ------
    ValueError
        If the covariance is not positively definite even with jitter, this error will be raised.
        Consider increasing the jitter for numerical stabilization.

    Notes
    -----
    If any NaN values are detected in `L`, an error message is logged, and a ValueError is raised,
    indicating that the covariance is not positively definite with the given jitter value.
    """
    sigma2 = square(sigma)
    sigma2 = where(sigma2 < jitter, jitter, sigma2)

    W = stabilize(cov_func(x, x), sigma2)
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
    x,
    cov_func,
    rank=DEFAULT_RANK,
    sigma=DEFAULT_SIGMA,
    jitter=DEFAULT_JITTER,
):
    R"""
    Compute a low rank :math:`L` such that :math:`L L^\top \approx K`, where :math:`K` is the
    full rank covariance matrix. The rank is less than or equal to the number of
    landmark points.

    Parameters
    ----------
    x : array-like
        The training instances.
    cov_func : function
        The Gaussian process covariance function.
    rank : int or float, optional
        The rank of the decomposition, or if rank is a float greater
        than 0 and less than 1, the eigenvalues of the included eigenvectors
        account for the specified percentage of the total eigenvalues.
        Defaults to 0.99.
    sigma : float, optional
        Noise standard deviation of the data we condition on. Defaults to 0.
    jitter : float, optional
        A small amount to add to the diagonal. Defaults to 1e-6.

    Returns
    -------
    array-like
        :math:`L` - A matrix such that :math:`L L^\top \approx K`.

    Notes
    -----
    The rank of the decomposition is determined by either the integer value provided or
    automatically selected to capture the specified percentage of total eigenvalues if a float is
    provided. This function computes the low-rank approximation of the full covariance matrix.
    """
    sigma2 = square(sigma)
    sigma2 = where(sigma2 < jitter, jitter, sigma2)

    W = stabilize(cov_func(x, x), sigma2)
    s, v = _eigendecomposition(W, rank=rank)
    L = v * sqrt(s)
    return L


def _standard_low_rank(
    x, cov_func, xu, Lp=None, sigma=DEFAULT_SIGMA, jitter=DEFAULT_JITTER
):
    R"""
    Compute a low rank :math:`L` such that :math:`L L^\top \approx K`,
    where :math:`K` is the full rank covariance matrix on `x`, and
    :math:`L_p L_p^\top = \Sigma_p` where :math:`\Sigma_p` is the full rank
    covariance matrix on `xu`. The rank is equal to the number of landmark points.

    Parameters
    ----------
    x : array-like
        The training instances.
    cov_func : function
        The Gaussian process covariance function.
    xu : array-like
        The landmark points.
    Lp : array-like, optional
        A matrix :math:`L_p L_p^\top = \Sigma_p` where :math:`\Sigma_p` is
        the full rank covariance matrix on the landmarks `xu`.
        Pass to avoid recomputing, by default None.
    sigma : float, optional
        Noise standard deviation of the data we condition on, by default 0.
    jitter : float, optional
        A small amount to add to the diagonal, by default 1e-6.

    Returns
    -------
    array-like, array-like
        :math:`L` - A matrix such that :math:`L L^\top \approx K`.
    """
    C = cov_func(x, xu)

    if Lp is None:
        Lp = _full_rank(xu, cov_func, sigma=sigma, jitter=jitter)
    L = solve_triangular(Lp, C.T, lower=True).T
    return L


def _modified_low_rank(
    x,
    cov_func,
    xu,
    rank=DEFAULT_RANK,
    sigma=DEFAULT_SIGMA,
    jitter=DEFAULT_JITTER,
):
    R"""
    Compute a low rank :math:`L` and :math:`L_p` such that :math:`L L^\top \approx K`,
    where :math:`K` is the full rank covariance matrix on `x`.
    The rank is equal to the number of landmark points. This is the improved
    Nyström rank reduction method.

    Parameters
    ----------
    x : array-like
        The training instances.
    cov_func : function
        The Gaussian process covariance function.
    xu : array-like
        The landmark points.
    rank : int or float, optional
        The rank of the decomposition, or if rank is a float
        0.0 :math:`\le` rank :math:`\le` 1.0, the rank is reduced further using
        the QR decomposition such that the eigenvalues of the included eigenvectors
        account for the specified percentage of the total eigenvalues. Defaults to 0.99.
    sigma : float, optional
        Noise standard deviation of the data we condition on. Defaults to 0.
    jitter : float, optional
        A small amount to add to the diagonal. Defaults to 1e-6.

    Returns
    -------
    array-like
        :math:`L` - A matrix such that :math:`L L^\top \approx K`.

    Notes
    -----
    This function computes a low-rank approximation of the full covariance matrix using
    an improved Nyström method. The rank reduction is controlled either by an integer value or
    a floating-point value that specifies the percentage of total eigenvalues.
    """
    sigma2 = square(sigma)
    sigma2 = where(sigma2 < jitter, jitter, sigma2)

    W = stabilize(cov_func(xu, xu), sigma2)
    C = cov_func(x, xu)
    Q, R = qr(C, mode="reduced")
    s, v = _eigendecomposition(W, rank=xu.shape[0])
    T = R @ v
    S, V = _eigendecomposition(T / s @ T.T, rank=rank)
    L = Q @ V * sqrt(S)
    return L
