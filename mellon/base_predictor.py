import sys
from importlib import import_module
from abc import ABC, abstractmethod
from datetime import datetime

import gzip
import bz2

import jax.numpy as jnp
import numpy as np
import json


class BasePredictor(ABC):
    @abstractmethod
    def __init__(self):
        """Initialize the predictor. Must be overridden by subclasses."""
        pass

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
        module_name = self.__class__.__module__
        module = import_module(module_name)
        version = module.get("__version__", "NA")
        state = {
            "data": self._data_dict(),
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
            val = jnp.array(value)
            setattr(self, name, val)

    def to_json(self, filename, compress=None):
        """Serialize the predictor to a JSON file.

        This method serializes the predictor to a JSON file. It can optionally
        compress the JSON file using gzip or bz2 compression.

        :param filename: The name of the JSON file to which to serialize the predictor.
        :type filename: str
        :param compress: The compression method to use ('gzip' or 'bz2'). If None, no compression is used.
        :type compress: str, optional
        """
        json_str = json.dumps(self.__getstate__())

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
        :rtype: BasePredictor subclass instance
        """
        if filename.endswith(".gz"):
            open_func = gzip.open
        elif filename.endswith(".bz2"):
            open_func = bz2.open
        else:
            open_func = open

        with open_func(filename, "rt") as f:
            json_str = f.read()

        state = json.loads(json_str)
        clsname = state["metadata"]["classname"]
        module_name = state["metadata"]["module"]

        module = import_module(module_name)
        Subclass = getattr(module, clsname)
        instance = Subclass.__new__(Subclass)
        instance.__setstate__(state)

        return instance
