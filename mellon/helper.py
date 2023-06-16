import functools
import inspect
from inspect import Parameter

from jax import jit, vmap

from jax.numpy import (
    atleast_2d,
    ndarray,
    array,
    isscalar,
    exp,
)

from .validation import _validate_array


def Exp(func):
    """
    Function wrapper, making a function that returns the exponent of the wrapped function.
    """

    def new_func(x):
        return exp(func(x))

    return new_func


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
            multi_time = _validate_array(multi_time, "multi_time")

            def at_time(t):
                return func(self, *args, **kwargs, time=t)

            if do_jit:
                at_time = jit(at_time)
            vfun = vmap(at_time, in_axes=0, out_axes=1)
            return vfun(multi_time)
        return func(self, *args, **kwargs)

    wrapper.__signature__ = new_sig
    return wrapper
