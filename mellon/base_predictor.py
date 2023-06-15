import sys
from importlib import import_module
from abc import ABC, abstractmethod
from datetime import datetime

import gzip
import bz2

import json

from .base_cov import Covariance
from .util import Log
from .helper import deserialize
from .derivatives import (
    derivative,
    gradient,
    hessian,
    hessian_log_determinant,
)
from .validation import _validate_time_x, _validate_float


logger = Log()


class Predictor(ABC):
    """
    Abstract base class for predictor models. It provides a common interface for all subclasses, which are expected to
    implement the `__call__` method for making predictions.

    An instance `predictor` of a subclass of `Predictor` can be used to make a prediction by calling it with input data `x`:

    >>> y = predictor(x)

    It is the responsibility of subclasses to define the behaviour of `_predict`.

    Methods
    -------
    __call__(x: Union[array-like, pd.DataFrame]):
        This makes predictions for an input `x`. The input data type can be either an array-like object
        (like list or numpy array) or a pandas DataFrame.

        Parameters
        ----------
        x : array-like or pandas.DataFrame
            The input data on which to make a prediction.

        Returns
        -------
        This method returns the predictions made by the model on the input data `x`.
        The prediction is usually array-like with as many entries as `x.shape[0]`.
    """

    @abstractmethod
    def __init__(self):
        """Initialize the predictor. Must be overridden by subclasses."""
        pass

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        string = (
            'A predictor of class "'
            + self.__class__.__name__
            + '" with covariance function "'
            + repr(self.cov_func)
            + '" and data:\n'
            + "\n".join(
                [
                    str(key) + ": " + repr(getattr(self, key))
                    for key in self._data_dict().keys()
                ]
            )
        )
        return string

    @abstractmethod
    def _predict(self, *args, **kwars):
        """Call the predictor. Must be overridden by subclasses."""

    def __call__(self, x):
        """Make a prediction based on input x.

        :param x: Input data to the predictor.
        :type x: array-like
        """
        return self._predict(x)

    @abstractmethod
    def _data_dict(self):
        """Return a dictionary containing the predictor's state data.
        All arrays nee to be numpy arrays for serialization.

        This method must be implemented by subclasses. It should return a
        dictionary where each key-value pair corresponds to an attribute of
        the predictor and its current state.

        :return: A dictionary containing the predictor's state data.
        :rtype: dict
        """
        pass

    def gradient(self, x, jit=True):
        R"""
        Conputes the gradient of the predict function for each line in x.

        :param x: Data points.
        :type x: array-like
        :param jit: Use jax just in time compilation. Defaults to True.
        :type jit: bool
        :return: gradiants - The gradient of function at each point in x.
            gradients.shape == x.shape
        :rtype: array-like
        """
        return gradient(self.__call__, x, jit=jit)

    def hessian(self, x, jit=True):
        R"""
        Conputes the hessian of the predict function for each line in x.

        :param x: Data points.
        :type x: array-like
        :param jit: Use jax just in time compilation. Defaults to True.
        :type jit: bool
        :return: hessians - The hessian matrix of function at each point in x.
            hessians.shape == X.shape + X.shape[1:]
        :rtype: array-like
        """
        return hessian(self.__call__, x, jit=jit)

    def hessian_log_determinant(self, x, jit=True):
        R"""
        Conputes the logarirhm of the determinat of the predict function for
        each line in x.

        :param x: Data points.
        :type x: array-like
        :param jit: Use jax just in time compilation. Defaults to True.
        :type jit: bool
        :return: signs, log_determinants - The sign of the determinant
            at each point x and the logarithm of its absolute value.
            signs.shape == log_determinants.shape == x.shape[0]
        :rtype: array-like, array-like
        """
        return hessian_log_determinant(self.__call__, x, jit=jit)

    def __getstate__(self):
        """Get the current state of the predictor for serialization.

        This method is called when serializing the predictor instance. It
        returns a dictionary that represents the current state of the
        predictor.

        :return: A dictionary representing the state of the predictor.
        :rtype: dict
        """
        module_name = self.__class__.__module__
        module = import_module(module_name)
        version = getattr(module, "__version__", "NA")

        state = {
            "data": self._data_dict(),
            "cov_func": self.cov_func.__getstate__(),
            "metadata": {
                "classname": self.__class__.__name__,
                "module_name": module_name,
                "module_version": version,
                "serialization_date": datetime.now().isoformat(),
                "python_version": sys.version,
            },
        }
        return state

    def __setstate__(self, state):
        """Set the current state of the predictor from deserialized state.

        This method is called when deserializing the predictor instance. It
        takes a dictionary that represents the state of the predictor and
        updates the predictor's attributes accordingly.

        :param state: A dictionary representing the state of the predictor.
        :type state: dict
        """
        data = state["data"]
        for name, value in data.items():
            val = deserialize(value)
            setattr(self, name, val)
        self.cov_func = Covariance.from_dict(state["cov_func"])

    def to_json(self, filename=None, compress=None):
        """Serialize the predictor to a JSON file.

        This method serializes the predictor to a JSON file. It can optionally
        compress the JSON file using gzip or bz2 compression.
        It automatically detects the compression method based on the file extension
        or use the compress keyword to determine the compression method.
        It also makes sure the file is saved with the appropriate file extension.

        :param filename: The name of the JSON file to which to serialize the predictor.
            If filname is None then the JSON string is returned instead.
        :type filename: str or None
        :param compress: The compression method to use ('gzip' or 'bz2'). If None, no compression is used.
        :type compress: str, optional
        """
        json_str = json.dumps(self.to_dict())

        if filename is None:
            return json_str

        if compress == "gzip":
            if isinstance(filename, str) and not filename.endswith(".gz"):
                filename += ".gz"
            with gzip.open(filename, "wt") as f:
                f.write(json_str)
        elif compress == "bz2":
            if isinstance(filename, str) and not filename.endswith(".bz2"):
                filename += ".bz2"
            with bz2.open(filename, "wt") as f:
                f.write(json_str)
        elif compress is None:
            with open(filename, "w") as f:
                f.write(json_str)
        else:
            msg = (
                f"Unknown compression format {compress}.\n"
                'Availabe formats are "gzip", "bz2" and None.'
            )
            logger.error(msg)
            raise ValueError(msg)
        logger.info(f"Written predictor to {filename}.")

    def to_dict(self):
        """Serialize the predictor to a python dictionary.

        :return: A python dictionary with the predictor data.
        :rtype: dict
        """
        return self.__getstate__()

    @classmethod
    def from_json(cls, filepath, compress=None):
        """Deserialize the predictor from a JSON file.

        This method deserializes the predictor from a JSON file. It automatically
        detects the compression method based on the file extension or uses
        the compress keyword to determine the compression method.

        :param filename: The path of the JSON file from which to deserialize the predictor.
        :type filename: str, pathlib.Path, or os.path
        :param compress: The compression method to use ('gzip' or 'bz2'). If None, no compression is used.
        :type compress: str, optional
        :return: An instance of the predictor.
        :rtype: Predictor subclass instance
        """
        filename = str(filepath)
        if compress is None:
            compress = "none"
        if compress == "gzip" or filename.endswith(".gz"):
            open_func = gzip.open
        elif compress == "bz2" or filename.endswith(".bz2"):
            open_func = bz2.open
        else:
            open_func = open

        with open_func(filepath, "rt") as f:
            json_str = f.read()

        return cls.from_json_str(json_str)

    @classmethod
    def from_dict(cls, data_dict):
        """Deserialize the predictor from a python dictionay.

        This method deserializes the predictor from a python dictionary.

        :param data_dict: The dictionary from which to deserialize the predictor.
        :type data_dict: dict
        :return: An instance of the predictor.
        :rtype: Predictor subclass instance
        """
        return cls.from_json_str(json.dumps(data_dict))

    @classmethod
    def from_json_str(cls, json_str):
        """Deserialize the predictor from a JSON string.

        This method deserializes the predictor from the content of a JSON file.

        :param json_str: The JSON string from which to deserialize the predictor.
        :type json_str: str
        :return: An instance of the predictor.
        :rtype: Predictor subclass instance
        """
        state = json.loads(json_str)
        clsname = state["metadata"]["classname"]
        module_name = state["metadata"]["module_name"]

        module = import_module(module_name)
        Subclass = getattr(module, clsname)
        instance = Subclass.__new__(Subclass)
        instance.__setstate__(state)

        return instance


class PredictorTime(Predictor):
    """
    Abstract base class for predictor models with a time covariate.

    An instance `predictor` of a subclass of `PredictorTime` can be used to
    make a prediction by calling it with input data `x` and `time`:

    >>> y = predictor(x, time)

    It is the responsibility of subclasses to define the behaviour of `_predict`.

    Methods
    -------
    __call__(x: Union[array-like, pd.DataFrame]):
        This makes predictions for an input `x`. The input data type can be either an array-like object
        (like list or numpy array) or a pandas DataFrame.

        Parameters
        ----------
        x : array-like or pandas.DataFrame
            The input data on which to make a prediction.
        time : scalar or array-like, optional
            The time points associated with each cell/row in 'x'.
            If 'time' is a scalar, it will be converted into a 1D array of the same size as 'x'.

        Returns
        -------
        This method returns the predictions made by the model on the input data `x` and `time`.
        The prediction is usually array-like with as many entries as `x.shape[0]`.
    """

    def __call__(self, Xnew, time=None):
        """
        Call method to use the class instance as a function. This method
        deals with an optional 'time' argument.
        If 'time' is a scalar, it converts it to a 1D array of the same size as 'Xnew'.

        Parameters
        ----------
        Xnew : array-like
            The new data points for prediction.
        time : scalar or array-like, optional
            The time points associated with each cell/row in 'Xnew'.
            If 'time' is a scalar, it will be converted into a 1D array of the same size as 'Xnew'.

        Returns
        -------
        array-like
            Predictions for 'Xnew'.

        Raises
        ------
        ValueError
            If 'time' is an array and its size does not match 'Xnew'.
        """

        # if time is a scalar, convert it into a 1D array of the same size as Xnew
        Xnew = _validate_time_x(Xnew, time, cast_scalar=True)

        return self._predict(Xnew)

    def time_derivative(self, x, time, jit=True):
        R"""
        Computes the time derivative of the prediction function for each line in `x`.

        This function applies a jax-based gradient operation to the density function evaluated at a specific time.
        The derivative is with respect to time and not the inputs in `x`.

        Parameters
        ----------
        x : array-like
            Data points where the derivative is to be evaluated.
        time : array-like or scalar
            Time point or points at which to evaluate the derivative.
            If time is an array then the time derivative will be computed for
            all data-points and all times in the array. It must be 1-d.
        time : array-like or scalar
            Time point or points at which to evaluate the derivative.
            If `time` is a scalar, the derivative will be computed at this
            specific time point for all data points in `x`.
            If `time` is an array, it should be 1-D and the time derivative
            will be computed for all data-points at the corresponding time in the array.
        jit : bool, optional
            If True, use JAX's just-in-time (JIT) compilation to speed up the computation. Defaults to True.

        Returns
        -------
        array-like
            The time derivative of the prediction function evaluated at each point in `x`.
            The shape of the output array is the same as `x`.

        """

        def dens_at(t):
            return self.__call__(x, t)

        return derivative(dens_at, time, jit=jit)

    def gradient(self, x, time, jit=True):
        R"""
        Conputes the gradient of the predict function for each cell state in x
        and one fixed time.

        :param x: Data points.
        :type x: array-like
        :param jit: Use jax just in time compilation. Defaults to True.
        :type jit: bool
        :return: gradiants - The gradient of function at each point in x.
            gradients.shape == x.shape
        :rtype: array-like
        """
        time = _validate_float(time, "time", optional=True)

        def dens_at(x):
            return self.__call__(x, time)

        return gradient(dens_at, x, jit=jit)

    def hessian(self, x, time, jit=True):
        R"""
        Conputes the hessian of the predict function for each line in x.

        :param x: Data points.
        :type x: array-like
        :param jit: Use jax just in time compilation. Defaults to True.
        :type jit: bool
        :return: hessians - The hessian matrix of function at each point in x.
            hessians.shape == X.shape + X.shape[1:]
        :rtype: array-like
        """
        time = _validate_float(time, "time", optional=True)

        def dens_at(x):
            return self.__call__(x, time)

        return hessian(dens_at, x, jit=jit)

    def hessian_log_determinant(self, x, time, jit=True):
        R"""
        Conputes the logarirhm of the determinat of the predict function for
        each line in x.

        :param x: Data points.
        :type x: array-like
        :param jit: Use jax just in time compilation. Defaults to True.
        :type jit: bool
        :return: signs, log_determinants - The sign of the determinant
            at each point x and the logarithm of its absolute value.
            signs.shape == log_determinants.shape == x.shape[0]
        :rtype: array-like, array-like
        """
        time = _validate_float(time, "time", optional=True)

        def dens_at(x):
            return self.__call__(x, time)

        return hessian_log_determinant(dens_at, x, jit=jit)
