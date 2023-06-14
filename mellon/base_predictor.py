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
    gradient,
    hessian,
    hessian_log_determinant,
)


logger = Log()


class Predictor(ABC):
    """
    Abstract base class for predictor models. It provides a common interface for all subclasses, which are expected to 
    implement the `__call__` method for making predictions.

    An instance `predictor` of a subclass of `Predictor` can be used to make a prediction by calling it with input data `x`:
    
    >>> y = predictor(x)

    It is the responsibility of subclasses to define the behaviour of `__call__`.

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
    def __call__(self, x):
        """Call the predictor on input x. Must be overridden by subclasses.

        :param x: Input data to the predictor.
        :type x: array-like
        """
        pass

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
