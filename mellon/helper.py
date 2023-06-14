from jax import jit, vmap

from jax.numpy import (
    atleast_2d,
    ndarray,
    array,
    isscalar,
)


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
    elif isinstance(x, slice):
        return {"type": "slice", "data": (x.start, x.stop, x.step)}
    elif isinstance(x, dict):
        return {"type": "dict", "data": {k: make_serializable(v) for k, v in x.items()}}
    else:
        return x


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
            return slice(*serializable_x["data"])
        elif data_type == "dict":
            return {k: deserialize(v) for k, v in serializable_x["data"].items()}
    else:
        return serializable_x


def vector_map(fun, X, in_axis=0, do_jit=True):
    """
    Applies jax just in time compilation and vmap to quickly evaluate a
    function for multiple input arrays.

    :param fun: The function to evaluate.
    :type fun: function
    :param X: Array of intput arrays.
    :type X: array-like
    :param in_axis: An integer, None, or (nested) standard Python container
        (tuple/list/dict) thereof specifying which input array axes to map over.
        S. documantation of jax.vmap.
    :return: Stacked results of the function calls.
    :rtype: array-like
    """
    if do_jit:
        fun = jit(fun)
    vfun = vmap(fun, in_axis)
    return vfun(X)


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
        x = x[:, active_dims]
    return x
