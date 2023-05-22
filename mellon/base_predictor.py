import sys
from importlib import import_module
from abc import ABC, abstractmethod
from datetime import datetime

import gzip
import bz2

from jax.numpy import asarray as asjnparray
import numpy as np
import json

from .base_cov import Covariance


class Predictor(ABC):
    @abstractmethod
    def __init__(self):
        """Initialize the predictor. Must be overridden by subclasses."""
        pass

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        string = (
            "A predictor of class \""
            + self.__class__.__name__
            + "\" with covariance function \""
            + repr(self.cov_func)
            + "\" and data:\n"
            + "\n".join([str(key) + ": " + repr(getattr(self, key)) for key in self._data_dict().keys()])
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

    def __getstate__(self):
        """Get the current state of the predictor for serialization.

        This method is called when serializing the predictor instance. It
        returns a dictionary that represents the current state of the
        predictor.

        :return: A dictionary representing the state of the predictor.
        :rtype: dict
        """
        module_name = self.__class__.__module__.split('.')[0]
        module = import_module(module_name)
        version = getattr(module, "__version__", "NA")
        state = {
            "data": self._data_dict(),
            "cov_func": self.cov_func.to_json(),
            "metadata": {
                "classname": self.__class__.__name__,
                "mellon_name": module_name,
                "mellon_version": version,
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
            val = asjnparray(value)
            setattr(self, name, val)
        self.cov_func = Covariance.from_json(state["cov_func"])

    def to_json(self, filename=None, compress=None):
        """Serialize the predictor to a JSON file.

        This method serializes the predictor to a JSON file. It can optionally
        compress the JSON file using gzip or bz2 compression.

        :param filename: The name of the JSON file to which to serialize the predictor.
            If filname is None then the JSON string is returned instead.
        :type filename: str or None
        :param compress: The compression method to use ('gzip' or 'bz2'). If None, no compression is used.
        :type compress: str, optional
        """
        json_str = json.dumps(self.__getstate__())

        if filename is None:
            return json_str

        if compress == "gzip":
            filename += ".gz"
            with gzip.open(filename, "wt") as f:
                f.write(json_str)
        elif compress == "bz2":
            filename += ".bz2"
            with bz2.open(filename, "wt") as f:
                f.write(json_str)
        else:
            with open(filename, "w") as f:
                f.write(json_str)

    @classmethod
    def from_json(cls, filename):
        """Deserialize the predictor from a JSON file.

        This method deserializes the predictor from a JSON file. It automatically
        detects the compression method based on the file extension.

        :param filename: The name of the JSON file from which to deserialize the predictor.
        :type filename: str
        :return: An instance of the predictor.
        :rtype: Predictor subclass instance
        """
        if filename.endswith(".gz"):
            open_func = gzip.open
        elif filename.endswith(".bz2"):
            open_func = bz2.open
        else:
            open_func = open

        with open_func(filename, "rt") as f:
            json_str = f.read()

        return cls.from_json_str(json_str)

    @classmethod
    def from_json_str(cls, json_str):
        """Deserialize the predictor from a JSON string.

        This method deserializes the predictor from a JSON file. It automatically
        detects the compression method based on the file extension.

        :param json_str: The JSON string from which to deserialize the predictor.
        :type json_str: str
        :return: An instance of the predictor.
        :rtype: Predictor subclass instance
        """
        state = json.loads(json_str)
        clsname = state["metadata"]["classname"]
        module_name = state["metadata"]["mellon_name"]

        module = import_module(module_name)
        Subclass = getattr(module, clsname)
        instance = Subclass.__new__(Subclass)
        instance.__setstate__(state)

        return instance
