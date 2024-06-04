import jax
from jax.numpy import isscalar, atleast_2d

from .validation import validate_1d, validate_float


def derivative(function, x, jit=True):
    """
    Computes the derivative of a scalar function at each point in `x`.

    This function applies a jax-based derivative operation to the input function evaluated at specific points in `x`.
    The derivative is with respect to the function's input.

    Parameters
    ----------
    function : callable
        A function that takes in a scalar input and outputs a scalar.
        The function must have the signature function(x: scalar) -> scalar.
    x : array-like or scalar
        Data point or points at which to evaluate the derivative.
        If `x` is an array then the derivative will be computed for
        all points in the array. It must be 1-d.
    jit : bool, optional
        If True, use JAX's just-in-time (JIT) compilation to speed up the computation. Defaults to True.

    Returns
    -------
    array-like
        The derivative of the function evaluated at each point in `x`.
        The shape of the output array is the same as `x`.

    """

    def get_grad(x):
        return jax.jacrev(function)(x)

    if isscalar(x):
        x = validate_float(x, "x")
        return get_grad(x)

    x = validate_1d(x)

    if jit:
        get_grad = jax.jit(get_grad)
    return jax.vmap(get_grad, in_axes=(0,))(x).T


def gradient(function, x, *args, jit=True):
    """
    Computes the gradient of a function for each line in `x`.

    Parameters
    ----------
    function : callable
        A function that takes a scalar input and outputs a scalar.
        The function must have the signature function(x: scalar) -> scalar.
    x : array-like
        Data points at which the gradient is to be computed.
    *args
        Additional arguments passed to `function`. shape[0] of all args shouold agree with x.shape[0].
    jit : bool, optional
        If True, use JAX's just-in-time (JIT) compilation to speed up the computation. Defaults to True.

    Returns
    -------
    array-like
        The gradient of the function at each point in `x`.
        The shape of the output array is the same as `x`.
    """

    def get_grad(x, *args):
        return jax.jacrev(function)(x[None, :], *args)

    if jit:
        get_grad = jax.jit(get_grad)
    in_axes = (0,) * (len(args) + 1)
    gradients = jax.vmap(get_grad, in_axes=in_axes)(x, *args)
    if len(gradients.shape) <= 4:
        return gradients.reshape(x.shape)
    return gradients.reshape(gradients.shape[::2])


def hessian(function, x, *args, jit=True):
    """
    Computes the gradient of a function for each line in `x`.

    Parameters
    ----------
    function : callable
        A function that takes a scalar input and outputs a scalar.
        The function must have the signature function(x: scalar) -> scalar.
    x : array-like
        Data points at which the gradient is to be computed.
    *args
        Additional arguments passed to `function`. shape[0] of all args shouold agree with x.shape[0].
    jit : bool, optional
        If True, use JAX's just-in-time (JIT) compilation to speed up the computation. Defaults to True.

    Returns
    -------
    array-like
        The gradient of the function at each point in `x`.
        The shape of the output array is the same as `x`.
    """
    x = atleast_2d(x)

    def get_hess(x, *args):
        return jax.jacfwd(jax.jacrev(function))(x[None, :], *args)

    if jit:
        get_hess = jax.jit(get_hess)
    out_shape = x.shape + x.shape[1:]
    in_axes = (0,) * (len(args) + 1)
    hessians = jax.vmap(get_hess, in_axes=in_axes)(x, *args)
    if len(hessians.shape) <= 6:
        return hessians.reshape(out_shape)
    return hessians.reshape(hessians.shape[::2])


def hessian_log_determinant(function, x, *args, jit=True):
    """
    Computes the logarithm of the determinant of the Hessian for each line in `x`.

    Parameters
    ----------
    function : callable
        A function that takes a scalar input and outputs a scalar.
        The function must have the signature function(x: scalar) -> scalar.
    x : array-like
        Data points at which the log determinant of the Hessian is to be computed.
    *args
        Additional arguments passed to `function`. shape[0] of all args shouold agree with x.shape[0].
    jit : bool, optional
        If True, use JAX's just-in-time (JIT) compilation to speed up the computation. Defaults to True.

    Returns
    -------
    array-like, array-like
        The sign of the determinant at each point in `x` and the logarithm of its absolute value.
        `signs.shape == log_determinants.shape == x.shape[0]`.
    """
    x = atleast_2d(x)

    d = x.shape[1]
    hess_shape = (d, d)

    def get_log_det(x, *args):
        hess = jax.jacfwd(jax.jacrev(function))(x[None, :], *args).reshape(hess_shape)
        sign, log_det = jax.numpy.linalg.slogdet(hess)
        return sign, log_det

    if jit:
        get_log_det = jax.jit(get_log_det)
    in_axes = (0,) * (len(args) + 1)
    return jax.vmap(get_log_det, in_axes=in_axes)(x, *args)
