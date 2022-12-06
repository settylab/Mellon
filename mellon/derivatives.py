import jax


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
