import sys
from abc import ABC, abstractmethod
from importlib import import_module
from datetime import datetime
import json

from jax.numpy import asarray as asjnparray

from .util import make_serializable


class Covariance(ABC):
    R"""
    Base covariance function.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def k(x, y):
        pass

    def __call__(self, x, y):
        return self.k(x, y)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(self, other)

    def __pow__(self, other):
        return Pow(self, other)

    def _data_dict(self):
        state = {key: make_serializable(val) for key, val in self.__dict__.items()}
        return state

    def __getstate__(self):
        """Get the current state of the covariance function for serialization.

        This method is called when serializing the predictor instance. It
        returns a dictionary that represents the current state of the
        predictor.

        :return: A dictionary representing the state of the predictor.
        :rtype: dict
        """
        module_name = self.__class__.__module__.split(".")[0]
        module = import_module(module_name)
        version = getattr(module, "__version__", "NA")
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
        """Set the current state of the covariance from deserialized state.

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

    def to_json(self):
        """Serialize the predictor to a JSON string.

        This method serializes the covariance function to a JSON string.

        :return: An JASON string.
        :rtype: string
        """
        return json.dumps(self.__getstate__())

    @classmethod
    def from_json(cls, json_str):
        """Deserialize the covariance function from a JSON string.

        This method deserializes the predictor from a JSON file. It automatically
        detects the compression method based on the file extension.

        :param json_str: The name of the JSON string from which to deserialize the covariance function.
        :type json_str: str
        :return: An instance of the covariance function.
        :rtype: Covariance subclass instance
        """

        state = json.loads(json_str)
        clsname = state["metadata"]["classname"]
        module_name = state["metadata"]["mellon_name"]

        if clsname in globals():
            Subclass = globals()[clsname]
        else:
            module = import_module(module_name)
            Subclass = getattr(module, clsname)
        instance = Subclass.__new__(Subclass)
        instance.__setstate__(state)

        return instance


class CovariancePair(Covariance):
    R"""
    Supports combining two covariance functions with.
    """

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return self.__repr__()

    @classmethod
    def __repr__(self):
        """Return a string representation"""
        pass

    @classmethod
    def k(self, x, y):
        """Implement the kernel that combines the two covariance functions
        self.left and self.right."""
        pass

    def __getstate__(self):
        """Get the current state of the covariance function for serialization.

        This method is called when serializing the covariance function instance. It
        returns a dictionary that represents the current state of the
        covariance function.

        :return: A dictionary representing the state of the covariance function.
        :rtype: dict
        """
        module_name = self.__class__.__module__.split(".")[0]
        module = import_module(module_name)
        version = getattr(module, "__version__", "NA")
        if callable(self.right):
            right_data = self.right.__getstate__()
        else:
            right_data = make_serializable(self.right)
        state = {
            "left_data": self.left.__getstate__(),
            "right_data": right_data,
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
        """Set the current state of the covariance from deserialized state.

        This method is called when deserializing the predictor instance. It
        takes a dictionary that represents the state of the covariance function and
        updates the functions's attributes accordingly.

        :param state: A dictionary representing the state of the covariance function.
        :type state: dict
        """
        data = state["data"]
        self.left = Covariance.from_json(data["left"])
        if isinstance(data["right"], dict):
            self.right = Covariance.from_json(data["right"])
        else:
            self.right = asjnparray(data["right"])


class Add(CovariancePair):
    R"""
    Supports adding covariance functions with + operator.
    """

    def __repr__(self):
        return "(" + repr(self.left) + " + " + repr(self.right) + ")"

    def k(self, x, y):
        if callable(self.right):
            return self.left(x, y) + self.right(x, y)
        return self.left(x, y) + self.right


class Mul(CovariancePair):
    R"""
    Supports multiplying covariance functions with * operator.
    """

    def __repr__(self):
        return "(" + repr(self.left) + " * " + repr(self.right) + ")"

    def k(self, x, y):
        if callable(self.right):
            return self.left(x, y) * self.right(x, y)
        return self.left(x, y) * self.right


class Pow(CovariancePair):
    R"""
    Supports taking a covariance function to a power with ** operator.
    """

    def __repr__(self):
        return "(" + repr(self.left) + " ** " + repr(self.right) + ")"

    def k(self, x, y):
        return self.left(x, y) ** self.right
