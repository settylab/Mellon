import sys
import logging
from jax import vmap, jacfwd
from jax.numpy import expand_dims, reshape
from abc import ABC, abstractmethod
from importlib import import_module
from datetime import datetime
import json

from .util import make_serializable, deserialize, select_active_dims, expand_to_inactive

MELLON_NAME = __name__.split(".")[0]

logger = logging.getLogger("mellon")


class Covariance(ABC):
    R"""
    Base covariance function.
    """

    def __init__(self, active_dims=None):
        self.active_dims = active_dims

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        """Return a string representation"""
        clsname = self.__class__.__name__
        arguments = [
            f"{key}={val}"
            for key, val in self.__dict__.items()
            if key != "active_dims" or val is not None
        ]
        return clsname + "(" + ", ".join(arguments) + ")"

    @abstractmethod
    def k(x, y):
        pass

    def k_grad(self, x):
        """
        Produce a function that computes the gradient of the kernel function
        with the left argument x with respect to y.
        This base class implementation uses jax automatic differentiation.
        Overwrite this method in the inheriting class if a more efficient
        implementation is available.

         Parameters
        ----------
        x : array-like
            First input array.

        Returns
        -------
        k_grad : callable
            Function that computes the gradient of the kernel function.
        """
        k_func = lambda y: self.k(x, y[None,])[..., 0]
        k_grad_pre = vmap(jacfwd(k_func), in_axes=(0,), out_axes=1)

        def k_grad(y):
            return k_grad_pre(y)

        return k_grad

    def __call__(self, x, y):
        return self.k(x, y)

    def diag(self, x):
        """
        Compute the diagonal of a covariance matrix.

        This function expands the input vectors, maps a function over them,
        reshapes the result and returns the diagonal of the covariance matrix.

        Parameters
        ----------
        x : ndarray
            Input array where the first dimension is the sample dimension and
            the second dimension corresponds to the state dimensions of the samples (cells).

        Returns
        -------
        diag : ndarray
            The diagonal of the covariance matrix.
        """

        x = expand_dims(x, 1)
        res = vmap(self.k)(x, x)
        diag = reshape(res, res.shape[:-2])
        return diag

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
        module_name = self.__class__.__module__
        clsname = self.__class__.__name__
        if module_name == "__main__":
            logger.warning(
                f'The covariance function "{clsname}" is not part of {MELLON_NAME} '
                f"and seems to be user defined. Make sure the implementation "
                "is available for deserialization."
            )
        elif module_name.split(".")[0] != MELLON_NAME:
            logger.warning(
                f'The covariance function "{clsname}" is not part of {MELLON_NAME} '
                f'but of the module "{module_name}". Make sure the module '
                "is available for deserialization."
            )
        metamodule = import_module(module_name.split(".")[0])
        module = import_module(module_name)
        metaversion = getattr(metamodule, "__version__", "NA")
        version = getattr(module, "__version__", metaversion)
        state = {
            "type": "mellon.Covariance",
            "data": self._data_dict(),
            "metadata": {
                "classname": clsname,
                "module_name": module_name,
                "module_version": version,
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
            val = deserialize(value)
            setattr(self, name, val)

    def to_json(self):
        """Serialize the predictor to a JSON string.

        This method serializes the covariance function to a JSON string.

        :return: An JASON string.
        :rtype: string
        """
        return json.dumps(self.__getstate__())

    def to_dict(self):
        """Serialize the predictor to a python dictionary."""
        return self.__getstate__()

    @classmethod
    def from_json(cls, json_str):
        """Deserialize the covariance function from a JSON string.

        This method deserializes the predictor from a JSON file. It automatically
        detects the compression method based on the file extension.

        :param json_str: The JSON string from which to deserialize the covariance function.
        :type json_str: str
        :return: An instance of the covariance function.
        :rtype: Covariance subclass instance
        """

        state = json.loads(json_str)
        return cls.from_dict(state)

    @classmethod
    def from_dict(cls, state):
        """Deserialize the covariance function from a python dictionary.

        :param state: The python dictionary from which to deserialize the covariance function.
        :type state: dict
        :return: An instance of the covariance function.
        :rtype: Covariance subclass instance
        """
        if not isinstance(state, dict) or state.get("type") != "mellon.Covariance":
            raise ValueError(
                "The passed dict does not seem to define a covariance kernel."
            )
        clsname = state["metadata"]["classname"]
        module_name = state["metadata"]["module_name"]

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

    def __init__(self, left, right, active_dims=None):
        super().__init__()
        self.left = left
        self.right = right
        self.active_dims = active_dims

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
            "type": "mellon.Covariance",
            "left_data": self.left.__getstate__(),
            "right_data": right_data,
            "active_dims": make_serializable(self.active_dims),
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
        """Set the current state of the covariance from deserialized state.

        This method is called when deserializing the predictor instance. It
        takes a dictionary that represents the state of the covariance function and
        updates the functions's attributes accordingly.

        :param state: A dictionary representing the state of the covariance function.
        :type state: dict
        """
        if not isinstance(state, dict) or state.get("type") != "mellon.Covariance":
            raise ValueError(
                "The passed dict does not seem to define a covariance kernel."
            )
        self.left = Covariance.from_dict(state["left_data"])
        if (
            isinstance(state["right_data"], dict)
            and state["right_data"].get("type") == "mellon.Covariance"
        ):
            self.right = Covariance.from_dict(state["right_data"])
        else:
            self.right = deserialize(state["right_data"])
        self.active_dims = deserialize(state.get("active_dims", None))


class Add(CovariancePair):
    R"""
    Supports adding covariance functions with + operator.
    """

    def __repr__(self):
        return "(" + repr(self.left) + " + " + repr(self.right) + ")"

    def k(self, x, y):
        x = select_active_dims(x, self.active_dims)
        y = select_active_dims(y, self.active_dims)

        if callable(self.right):
            return self.left(x, y) + self.right(x, y)
        return self.left(x, y) + self.right

    def k_grad(self, x):
        """
        Generate a function to compute the gradient of the sum of covariance kernels.

        This method returns a callable that, when given an array `y`, computes the gradient
        with respect to `y`, considering `x` as the fixed input. The computation is
        restricted to the active dimensions specified in the covariance function instance.

        Parameters
        ----------
        x : array-like
            The fixed input array used as the first argument in the Linear kernel.
            Its shape should be compatible with the active dimensions of the kernel.

        Returns
        -------
        Callable
            A function that takes an array `y` as input and returns the gradient of the
            Linear kernel function with respect to `y`, evaluated at the pair `(x, y)`.
            The gradient is computed only over the active dimensions.
        """
        x_shape = x.shape
        active_dims = self.active_dims
        x = select_active_dims(x, active_dims)
        left_grad = self.left.k_grad(x)

        if callable(self.right):
            right_grad = self.right.k_grad(x)

            def k_grad(y):
                y_shape = y.shape
                y = select_active_dims(y, active_dims)
                grad = left_grad(y) + right_grad(y)
                target_shape = x_shape[:-1] + y_shape
                full_grad = expand_to_inactive(grad, target_shape, active_dims)
                return full_grad

        else:

            def k_grad(y):
                y_shape = y.shape
                y = select_active_dims(y, self.active_dims)
                grad = left_grad(y)
                target_shape = x_shape[:-1] + y_shape
                full_grad = expand_to_inactive(grad, target_shape, active_dims)
                return full_grad

        return k_grad


class Mul(CovariancePair):
    R"""
    Supports multiplying covariance functions with * operator.
    """

    def __repr__(self):
        return "(" + repr(self.left) + " * " + repr(self.right) + ")"

    def k(self, x, y):
        x = select_active_dims(x, self.active_dims)
        y = select_active_dims(y, self.active_dims)

        if callable(self.right):
            return self.left(x, y) * self.right(x, y)
        return self.left(x, y) * self.right

    def k_grad(self, x):
        """
        Generate a function to compute the gradient of the product of covariance kernels.

        This method returns a callable that, when given an array `y`, computes the gradient
        with respect to `y`, considering `x` as the fixed input. The computation is
        restricted to the active dimensions specified in the covariance function instance.

        Parameters
        ----------
        x : array-like
            The fixed input array used as the first argument in the covariance function.
            Its shape should be compatible with the active dimensions of the kernel.

        Returns
        -------
        Callable
            A function that takes an array `y` as input and returns the gradient of the
            product kernel function with respect to `y`, evaluated at the pair `(x, y)`.
            The gradient is computed only over the active dimensions.
        """
        x_shape = x.shape
        active_dims = self.active_dims
        x = select_active_dims(x, active_dims)
        left_grad_func = self.left.k_grad(x)

        if callable(self.right):
            right_grad_func = self.right.k_grad(x)

            def k_grad(y):
                y_shape = y.shape
                y = select_active_dims(y, active_dims)

                left_k = self.left.k(x, y)[..., None]
                right_k = self.right.k(x, y)[..., None]
                left_grad = left_grad_func(y)
                right_grad = right_grad_func(y)

                grad = left_grad * right_k + left_k * right_grad
                target_shape = x_shape[:-1] + y_shape
                full_grad = expand_to_inactive(grad, target_shape, active_dims)
                return full_grad

        else:

            def k_grad(y):
                y_shape = y.shape
                y = select_active_dims(y, active_dims)
                left_grad = left_grad_func(y)

                grad = left_grad * self.right
                target_shape = x_shape[:-1] + y_shape
                full_grad = expand_to_inactive(grad, target_shape, active_dims)
                return full_grad

        return k_grad


class Pow(CovariancePair):
    R"""
    Supports taking a covariance function to a power with ** operator.
    """

    def __repr__(self):
        return "(" + repr(self.left) + " ** " + repr(self.right) + ")"

    def k(self, x, y):
        x = select_active_dims(x, self.active_dims)
        y = select_active_dims(y, self.active_dims)

        return self.left(x, y) ** self.right

    def k_grad(self, x):
        """
        Generate a function to compute the gradient of the covariance kernels to a power.

        This method returns a callable that, when given an array `y`, computes the gradient
        with respect to `y`, considering `x` as the fixed input. The computation is
        restricted to the active dimensions specified in the covariance function instance.

        Parameters
        ----------
        x : array-like
            The fixed input array used as the first argument in the covariance function.
            Its shape should be compatible with the active dimensions of the kernel.

        Returns
        -------
        Callable
            A function that takes an array `y` as input and returns the gradient of the
            product kernel function with respect to `y`, evaluated at the pair `(x, y)`.
            The gradient is computed only over the active dimensions.
        """
        x_shape = x.shape
        active_dims = self.active_dims
        x = select_active_dims(x, active_dims)

        # Obtain the gradient function for the base covariance function
        base_grad_func = self.left.k_grad(x)

        def k_grad(y):
            y_shape = y.shape
            y = select_active_dims(y, active_dims)
            base_k = self.left.k(x, y)[..., None]
            base_grad = base_grad_func(y)

            # Compute the gradient of the powered covariance function using the chain rule
            # (f(x)^n)' = n * f(x)^(n-1) * f'(x)
            power_grad = self.right * (base_k ** (self.right - 1)) * base_grad

            target_shape = x_shape[:-1] + y_shape
            full_grad = expand_to_inactive(power_grad, target_shape, active_dims)
            return full_grad

        return k_grad
