import sys
import logging
import functools
import inspect
from typing import List
from inspect import Parameter
from enum import Enum
from itertools import islice

from jax import config as jaxconfig
from jax.numpy import (
    eye,
    log,
    pi,
    zeros,
    repeat,
    newaxis,
    tensordot,
    sqrt,
    maximum,
    triu_indices,
    sort,
    ones,
    arange,
    concatenate,
    atleast_2d,
    ndarray,
    array,
    isscalar,
    where,
    vstack,
    index_exp,
    broadcast_to,
    integer,
    floating,
)
from jax.numpy import sum as arraysum
from jax.numpy import diag as diagonal
from jax.numpy.linalg import norm, lstsq, matrix_rank
from jax.scipy.special import gammaln
from jax import vmap, jit
from sklearn.neighbors import BallTree, KDTree

from .validation import validate_array

logger = logging.getLogger("mellon")

DEFAULT_JITTER = 1e-6
DEFAULT_RANK_TOL = 5e-1


def batched_vmap(func, x, *args, batch_size=100):
    """Apply function in batches to save memory."""
    n = x.shape[0]
    out_list = []
    it = iter(range(n))
    while idx := tuple(islice(it, batch_size)):
        batch = x[idx, ...]
        out_list.append(vmap(func, in_axes=(0, None))(batch, *args))
    return vstack(out_list)


def _None_to_str(v):
    if v is None:
        return "None"
    return v


def make_serializable(x):
    """
    Convert the input into a serializable format.

    Parameters
    ----------
    x : variable
        The input variable that can be array-like, slice, scalar or dict.

    Returns
    -------
    serializable_x : variable
        The input variable converted into a serializable format.
    """
    if isinstance(x, ndarray):
        return {"type": "jax.numpy", "data": x.tolist()}
    if isinstance(x, integer):
        return int(x)
    if isinstance(x, floating):
        return float(x)
    elif isinstance(x, slice):
        dat = [_None_to_str(v) for v in (x.start, x.stop, x.step)]
        return {"type": "slice", "data": dat}
    elif isinstance(x, dict):
        return {"type": "dict", "data": {k: make_serializable(v) for k, v in x.items()}}
    elif isinstance(x, set):
        return {"type": "set", "data": [make_serializable(v) for v in x]}
    else:
        return _None_to_str(x)


def _str_to_None(v):
    if isinstance(v, str) and v == "None":
        return None
    return v


def deserialize(serializable_x):
    """
    Convert the serializable input back into the original format.

    Parameters
    ----------
    serializable_x : variable
        The input variable that is in a serializable format.

    Returns
    -------
    x : variable
        The input variable converted back into its original format.
    """
    if isinstance(serializable_x, dict):
        data_type = serializable_x["type"]
        if data_type == "jax.numpy":
            return array(serializable_x["data"])
        elif data_type == "slice":
            dat = [_str_to_None(v) for v in serializable_x["data"]]
            return slice(*dat)
        elif data_type == "dict":
            return {k: deserialize(v) for k, v in serializable_x["data"].items()}
        elif data_type == "set":
            return {deserialize(v) for v in serializable_x["data"]}
    else:
        return _str_to_None(serializable_x)


def ensure_2d(X):
    """
    Ensures that the input JAX array, X, is at least 2-dimensional.

    :param X: The input JAX array to be made 2-dimensional.
    :type X: jnp.array
    :return: The input array transformed to a 2-dimensional array.
    :rtype: jnp.array

    If X is 1-dimensional, it is reshaped to a 2-dimensional array,
    where each element of X becomes a row in the 2-dimensional array.
    """
    return atleast_2d(X.T).T


def select_active_dims(x, active_dims):
    """
    Select the active dimensions from the input.

    Parameters
    ----------
    x : array-like
        Input array.

    selected_dimensions : array-like, slice or scalar
        The indices of the active dimensions. It could be a scalar, a list, a numpy array, or a slice object.

    Returns
    -------
    x : array-like
        Array with selected dimensions.
    """
    if active_dims is not None:
        if isscalar(active_dims):
            active_dims = [active_dims]
        x = x[..., active_dims]
    return x


def expand_to_inactive(values, target_shape, active_dims):
    """
    Expand the values to the full target shape with zeros in inactive dimensions,
    specifically targeting the last dimension based on active_dims.

    Parameters
    ----------
    values : jax.numpy.ndarray
        Gradient values corresponding to the active dimensions.
    target_shape : tuple
        The shape of the full gradient array to be constructed.
    active_dims : slice
        Specifies the active dimensions as a slice object.

    Returns
    -------
    jax.numpy.ndarray
        An array with the specified target shape, where values are set in active dimensions
        of the last dimension, and zeros are placed elsewhere.
    """
    if active_dims is None:
        return values

    if isscalar(active_dims):
        active_dims = [active_dims]

    full_array = zeros(target_shape, dtype=values.dtype)
    full_array = full_array.at[..., active_dims].set(values)

    return full_array


def make_multi_time_argument(func):
    """
    Decorator to modify a method to optionally take a multi-time argument.

    This decorator modifies the method it wraps to take an optional `multi_time` keyword argument.
    If `multi_time` is provided, the decorated method will be called once for each value in `multi_time`
    with that value passed as the `time` argument to the method.

    The original method's signature and docstring are preserved.

    Parameters
    ----------
    func : callable
        The method to be modified. This method must take a `time` keyword argument.

    Returns
    -------
    callable
        The modified method.

    Examples
    --------
    .. code-block:: python

        class MyClass:
            @make_multi_time_argument
            def method(self, x, time=None):
                return x + time

        my_object = MyClass()
        print(my_object.method(1, multi_time=np.array([1, 2, 3])))
        # Output: array([2, 3, 4])
    """
    sig = inspect.signature(func)
    new_params = list(sig.parameters.values()) + [
        Parameter("multi_time", Parameter.POSITIONAL_OR_KEYWORD, default=None)
    ]
    new_sig = sig.replace(parameters=new_params)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        multi_time = kwargs.pop("multi_time", None)
        do_jit = kwargs.get("jit", False)
        if multi_time is not None:
            if kwargs.get("time", None) is not None:
                raise ValueError(
                    "Cannot specify both 'time' and 'multi_time' arguments"
                )
            multi_time = validate_array(multi_time, "multi_time")

            def at_time(t):
                return func(self, *args, **kwargs, time=t)

            if do_jit:
                at_time = jit(at_time)
            vfun = vmap(at_time, in_axes=0, out_axes=1)
            return vfun(multi_time)
        return func(self, *args, **kwargs)

    wrapper.__signature__ = new_sig
    return wrapper


def add_diagonal(A, value):
    R"""
    Add a value to the diagonal of matrix :math:`A`.

    :param A: A square matrix.
    :param value: The amount to add to the diagonal.
    :type value: float
    :return: :math:`A'` - The matrix :math:`A` with a value added to the diagonal.
    :rtype: array-like
    """
    n = A.shape[0]
    return A + eye(n) * value


def stabilize(A, jitter=DEFAULT_JITTER):
    R"""
    Add a small jitter to the diagonal for numerical stability.

    :param A: A square matrix.
    :param jitter: The amount to add to the diagonal. Defaults to 1e-6.
    :type jitter: float
    :return: :math:`A'` - The matrix :math:`A` with a small jitter added to the diagonal.
    :rtype: array-like
    """
    return add_diagonal(A, jitter)


def add_variance(K, M=None, jitter=DEFAULT_JITTER):
    R"""
    Computes :math:`K + MM^T` where the diagonal of :math:`MM^T` is
    at least `jitter`. This function stabilizes :math:`K` for the
    Cholesky decomposition if not already stable enough through adding :math:`MM^T`.

    Parameters
    ----------
    K : array_like, shape (n, n)
        A covariance matrix.
    M : array_like, shape (n, p), optional
        Left factor of additional variance. Default is 0.
    jitter : float, optional
        A small number to stabilize the covariance matrix. Defaults to 1e-6.

    Returns
    -------
    combined_covariance : array_like, shape (n, n)
        A combined covariance matrix that is more stably positive definite.

    Notes
    -----
    If `M` is None, the function will add the jitter to the diagonal of `K` to
    make it more stable. Otherwise, it will add :math:`MM^T` and correct the
    diagonal based on the `jitter` parameter.
    """
    if M is None:
        K = stabilize(K, jitter=jitter)
    elif isscalar(M):
        sigma2 = max(jitter, M**2)
        K = add_diagonal(K, sigma2)
    else:
        noise = M.dot(M.T)
        diff = where(diagonal(noise) < jitter, jitter - diagonal(noise), 0)
        K = K + noise + diagonal(diff)
    return K  # Ensure the combined matrix is positive definite


def mle(nn_distances, d):
    R"""
    Nearest Neighbor distribution maximum likelihood estimate for log density
    given observed nearest neighbor distances :math:`nn\text{_}distances` in
    dimensions :math:`d`: :math:`mle = \log(\text{gamma}(d/2 + 1)) - (d/2) \cdot \log(\pi) -
    d \cdot \log(nn\text{_}distances)`

    :param nn_distances: The observed nearest neighbor distances.
    :type nn_distances: array-like
    :param d: The local dimensionality of the data.
    :type d: int
    :return: :math:`mle` - The maximum likelihood estimate at each point.
    :rtype: array-like
    """
    return gammaln(d / 2 + 1) - (d / 2) * log(pi) - d * log(nn_distances)


def distance(x, y):
    """
    Computes the distances between each point in x and y.

    :param x: A set of points.
    :type x: array-like
    :param y: A set of points.
    :type y: array-like
    :return: distances - The distance between each point in x and y.
    :rtype: array-like
    """
    xx = arraysum(x * x, axis=1)[:, newaxis]
    yy = arraysum(y * y, axis=1)[newaxis, :]
    xy = tensordot(x, y, (1, 1))
    sq = xx - 2 * xy + yy + 1e-12
    return sqrt(maximum(sq, 0))


def distance_grad(x, eps=1e-12):
    """
    Produces a function that computes the Euclidean distance from a fixed set of points `x`
    to any other set of points `y`, and the gradient of this distance with respect to `y`.

    Parameters
    ----------
    x : array-like
        A fixed set of points. The array should have a shape of (n, d), where `n` is the number
        of points and `d` is the dimensionality of each point.
    eps : float, optional
        A small epsilon value added to the distance to avoid division by zero when computing
        the gradient. Default is 1e-12.

    Returns
    -------
    function
        A function that, when called with an array-like `y` of shape (m, d), returns a tuple
        containing:

        - **distance** : ndarray
            An array of shape (n, m) representing the Euclidean distances from each point in `x`
            to each point in `y`.

        - **gradient** : ndarray
            An array of shape (n, m, d) representing the gradient of the distance with respect to
            each point in `y`.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[0, 0], [1, 1]])
    >>> dist_grad_func = distance_grad(x)
    >>> y = np.array([[1, 0], [0, 1]])
    >>> distance, gradient = dist_grad_func(y)
    >>> distance.shape
    (2, 2)
    >>> gradient.shape
    (2, 2, 2)

    Notes
    -----
    The gradient is computed using the formula for the derivative of the Euclidean distance
    with respect to the points `y`. The epsilon value is added to the computed squared distances
    before taking the square root to ensure numerical stability.
    """
    xx = arraysum(x * x, axis=1)[:, newaxis]

    def grad(y):
        yy = arraysum(y * y, axis=1)[newaxis, :]
        xy = tensordot(x, y, axes=(1, 1))
        sq = xx - 2 * xy + yy + eps
        distance = sqrt(maximum(sq, 0))
        delta = y[newaxis, :] - x[:, newaxis]
        gradient = delta / (distance[..., newaxis] + eps)
        return distance, gradient

    return grad


def test_rank(input, tol=DEFAULT_RANK_TOL, threshold=None):
    """
    Inspects the approximate rank of the transformation matrix L. The input can be the matrix itself or an object
    containing L as an attribute. A high rank indicates a potentially insufficient latent representation,
    suggesting a need for a more complex transformation matrix. Also allows logging based on a rank fraction threshold.

    :param input: The matrix L or an object containing it as an attribute.
    :type input: array-like or mellon estimator object
    :param tol: The rank calculation tolerance, defaults to {DEFAULT_RANK_TOL}.
    :type tol: float, optional
    :param threshold: If provided, logs a message based on the rank fraction.
    :type threshold: float, optional
    :return: The approximate rank of the matrix.
    :rtype: int
    :raises ValueError: If the input matrix is not 2D.
    """
    if hasattr(input, "shape"):
        L = input
    elif hasattr(input, "L"):
        L = input.L
        if L is None:
            raise AttributeError(
                "Matrix L is not found in the estimator object. Consider running `.prepare_inference()`."
            )
    else:
        raise TypeError(
            "Input must be either a matrix or a mellon enstimator with a transformation L."
        )

    if len(L.shape) != 2:
        raise ValueError("Matrix L must be 2D.")

    approx_rank = matrix_rank(L, rtol=tol)
    max_rank = min(L.shape)
    rank_fraction = approx_rank / max_rank

    if threshold is not None:
        if rank_fraction > threshold:
            logger.warning(
                f"High approx. rank fraction ({rank_fraction:.1%}). "
                "Consider increasing 'n_landmarks'."
            )
        else:
            logger.info(
                f"Rank fraction ({rank_fraction:.1%}, lower is better) is "
                "within acceptable range. "
                "Current settings should provide satisfactory model performance."
            )
    else:
        print(
            f"The approx. rank fraction is {rank_fraction:.1%} "
            f"({approx_rank:,} of {max_rank:,}). Lower is better."
        )

    return approx_rank.item()


def local_dimensionality(x, k=30, x_query=None, neighbor_idx=None):
    """
    Compute an estimate of the local fractal dimension of a data set using nearest neighbors.

    :param x: The input samples.
    :type x: array-like of shape (n_samples, n_features)
    :param k: The number of neighbors to consider, defaults to 30.
    :type k: int, optional
    :param x_query: The points at which to compute the local fractal dimension.
        If None, use x itself, defaults to None.
    :type x_query: array-like of shape (n_queries, n_features), optional
    :param neighbor_idx: The indices of the neighbors for each query point.
        If None, these are computed using a nearest neighbor search, defaults to None.
    :type neighbor_idx: array-like of shape (n_queries, k), optional
    :return: The estimated local fractal dimension at each query point.
    :rtype: array-like of shape (n_queries,)

    This function computes the local fractal dimension of a dataset at query points.
    It uses nearest neighbors and fits a line in log-log space to estimate the fractal dimension.
    """
    if k > x.shape[0]:
        logger.warning(
            f"Number of nearest neighbors (k={k}) is "
            f"greater than the number of samples ({x.shape[0]}). "
            "Setting k to the number of samples."
        )
        k = x.shape[0]
    if neighbor_idx is None:
        if x_query is None:
            x_query = x
        if x.shape[1] >= 20:
            tree = BallTree(x, metric="euclidean")
        else:
            tree = KDTree(x, metric="euclidean")
        neighbors = x[tree.query(x_query, k=k)[1]]
    else:
        neighbors = x[neighbor_idx]
    i, j = triu_indices(k, k=1)
    neighbor_distances = norm(neighbors[..., i, :] - neighbors[..., j, :], axis=-1)
    neighborhood_distances = sort(neighbor_distances, axis=-1)

    kc2 = k * (k - 1) // 2
    A = concatenate(
        [log(neighborhood_distances)[..., None], ones((x_query.shape[0], kc2, 1))],
        axis=-1,
    )
    y = log(arange(1, kc2 + 1))[:, None]

    vreg = vmap(lstsq, in_axes=(0, None))
    w, _, _, _ = vreg(A, y)
    return w[:, 0, 0]


def set_verbosity(verbose: bool):
    """
    Adjusts the verbosity of mellon logging.

    Parameters
    ----------
    verbose : bool
        If True, sets the logging level to INFO to show detailed logs.
        If False, sets it to WARNING to show only important messages.

    Notes
    -----
    This function provides a simplified interface for controlling logging
    verbosity, making it more accessible to users who are not familiar
    with the logging module's levels. You can access the mellon logger
    through `mellon.logger` to perform logging operations throughout the
    mellon module.

    Example
    -------
    To enable detailed logging, showing more information:

    >>> set_verbosity(True)

    To reduce the log output to only include warnings and errors:

    >>> set_verbosity(False)
    """
    level = logging.INFO if verbose else logging.WARNING
    logger.setLevel(level)
    logger.info(f"Logging verbosity set to {'INFO' if verbose else 'WARNING'}.")


def set_jax_config(enable_x64=True, platform_name="cpu"):
    """
    Sets up the JAX configuration with the specified settings.

    Parameters
    ----------
    enable_x64 : bool, optional
        Whether to enable 64-bit (double precision) computations in JAX.
        Defaults to True.
    platform_name : str, optional
        The platform name to use in JAX ('cpu', 'gpu', or 'tpu').
        Defaults to 'cpu'.
    """
    jaxconfig.update("jax_enable_x64", enable_x64)
    jaxconfig.update("jax_platform_name", platform_name)


class GaussianProcessType(Enum):
    """
    Defines types of Gaussian Process (GP) computations for various estimators within the mellon library:
    :class:`mellon.model.DensityEstimator`, :class:`mellon.model.FunctionEstimator`,
    :class:`mellon.model.DimensionalityEstimator`, :class:`mellon.model.TimeSensitiveDensityEstimator`.

    This enum can be passed through the `gp_type` attribute to the mentioned estimators.
    If a string representing one of these values is passed alternatively, the
    :func:`from_string` method is called to convert it to a `GaussianProcessType`.

    Options are 'full', 'full_nystroem', 'sparse_cholesky', 'sparse_nystroem', 'fixed'.
    """

    FULL = "full"
    FULL_NYSTROEM = "full_nystroem"
    SPARSE_CHOLESKY = "sparse_cholesky"
    SPARSE_NYSTROEM = "sparse_nystroem"
    FIXED = "fixed"

    @staticmethod
    def from_string(s: str, optional: bool = False):
        """
        Converts a string to a GaussianProcessType object or raises an error.

        Parameters
        ----------
        s : str
            The type of Gaussian Process (GP). Options are:
             - 'full': None-sparse GP
             - 'full_nystroem': Sparse GP with Nyström rank reduction
             - 'sparse_cholesky': Sparse GP using landmarks/inducing points
             - 'sparse_nystroem': Sparse GP along with an improved Nyström rank reduction
             - 'fixed': Using landmarks/inducing points even if there are less data points
        optional : bool, optional
            Specifies if the input is optional. Returns None if True and input is None.

        Returns
        -------
        GaussianProcessType
            Corresponding Gaussian Process type.

        Raises
        ------
        ValueError
            If the input does not correspond to any known Gaussian Process type.
        """

        if s is None:
            if optional:
                return None
            else:
                logger.error("Gaussian process type must be specified but is None.")
                raise ValueError("Gaussian process type must be specified but is None.")

        if isinstance(s, GaussianProcessType):
            return s

        normalized_input = s.lower().replace(" ", "_")

        # Try to match the exact Enum value
        for gp_type in GaussianProcessType:
            if gp_type.value == normalized_input:
                logger.info(f"Gaussian Process type: {gp_type.value}")
                return gp_type

        # If no exact match, try partial matching by finding the closest match
        for gp_type in GaussianProcessType:
            if normalized_input in gp_type.value:
                logger.warning(
                    f"Partial match found for Gaussian Process type: {gp_type.value}. Input was: {s}"
                )
                return gp_type

        message = f"Unknown Gaussian Process type: {s}"
        logger.error(message)
        raise ValueError(message)


def object_str(obj: object, dim_names: List[str] = None) -> str:
    """
    Generate a concise string representation of metadata for array-like objects.

    Parameters
    ----------
    obj : object
        Object for which to generate metadata string.

    dim_names : list of str, optional
        Names for dimensions, used for array-like objects.

    Returns
    -------
    str
        Metadata string.

    Examples
    --------
    >>> object_metadata_str(np.array([[1, 2], [3, 4]]), dim_names=['row', 'col'])
    '<array 2 row x 2 col, dtype=int64>'

    >>> object_metadata_str(np.array([1, 2, 3]), dim_names=['element'])
    '<array 3 element, dtype=int64>'

    >>> object_metadata_str("hello")
    'hello'
    """
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        dims = obj.shape
        if dim_names:
            dim_strs = [f"{dim:,} {name}" for dim, name in zip(dims, dim_names)]
        else:
            dim_strs = [f"{dim:,}" for dim in dims]

        for i in range(len(dim_strs), len(dims)):
            dim_strs.append(f"{dims[i]} dimension {i + 1}")

        dim_str = " x ".join(dim_strs)
        return f"<array {dim_str}, dtype={obj.dtype}>"
    else:
        return str(obj)


def object_html(obj: object, dim_names: list = None) -> str:
    """
    Generate an HTML string representation of metadata for array-like objects.

    Parameters
    ----------
    obj : object
        Object for which to generate metadata HTML.

    dim_names : list of str, optional
        Names for dimensions, used for array-like objects.

    Returns
    -------
    str
        HTML string with metadata.
    """

    def escape_html(text):
        """Escape HTML special characters."""
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        dims = obj.shape
        dim_count = len(dims)
        dim_names = dim_names or []

        # Format dimensions
        dim_strs = [
            f"{dim:,} {name}" if name else f"{dim:,}"
            for dim, name in zip(
                dims, dim_names + [None] * (dim_count - len(dim_names))
            )
        ]

        dim_str = " x ".join(dim_strs)
        return f"<span>&lt;array {escape_html(dim_str)}, dtype={escape_html(obj.dtype)}&gt;</span>"

    else:
        # Handle string representation with escaping
        return f"<span>{escape_html(obj)}</span>"
