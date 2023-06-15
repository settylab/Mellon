import jax

from .validation import _validate_1d


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
    x = _validate_1d(x)

    def get_grad(x):
        return jax.jacrev(function)(x).squeeze()

    if jit:
        get_grad = jax.jit(get_grad)
    return jax.vmap(get_grad, in_axes=(0,))(x)


def gradient(function, x, jit=True):
    R"""
    Conputes the gradient of the function for each line in x.

    :param function: A function returning one value per
        line in x. function(x).shape == (x.shape[0], )
    :type function: function
    :param x: Data points.
    :type x: array-like
    :param jit: Use jax just in time compilation. Defaults to True.
    :type jit: bool
    :return: gradiants - The gradient of function at each point in x.
        gradients.shape == x.shape
    :rtype: array-like
    """

    def get_grad(x):
        return jax.jacrev(function)(x[None, :]).squeeze()

    if jit:
        get_grad = jax.jit(get_grad)
    return jax.vmap(get_grad, in_axes=(0,))(x)


def hessian(function, x, jit=True):
    R"""
    Conputes the hessian of the function for each line in x.

    :param function: A function returning one value per
        line in x. function(x).shape == (x.shape[0], )
    :type function: function
    :param x: Data points.
    :type x: array-like
    :param jit: Use jax just in time compilation. Defaults to True.
    :type jit: bool
    :return: hessians - The hessian matrix of function at each point in x.
        hessians.shape == X.shape + X.shape[1:]
    :rtype: array-like
    """

    def get_hess(x):
        return jax.jacfwd(jax.jacrev(function))(x[None, :]).squeeze()

    if jit:
        get_hess = jax.jit(get_hess)
    return jax.vmap(get_hess, in_axes=(0,))(x)


def hessian_log_determinant(function, x, jit=True):
    R"""
    Conputes the logarirhm of the determinat of the function for each line in x.

    :param function: A function returning one value per
        line in x. function(x).shape == (x.shape[0], )
    :type function: function
    :param x: Data points.
    :type x: array-like
    :param jit: Use jax just in time compilation. Defaults to True.
    :type jit: bool
    :return: signs, log_determinants - The sign of the determinant
        at each point x and the logarithm of its absolute value.
        signs.shape == log_determinants.shape == x.shape[0]
    :rtype: array-like, array-like
    """

    def get_log_det(x):
        hess = jax.jacfwd(jax.jacrev(function))(x[None, :]).squeeze()
        sign, log_det = jax.numpy.linalg.slogdet(hess)
        return sign, log_det

    if jit:
        get_log_det = jax.jit(get_log_det)
    return jax.vmap(get_log_det, in_axes=(0,))(x)
