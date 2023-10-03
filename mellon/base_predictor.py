import sys
import logging
from importlib import import_module
from packaging import version
from abc import ABC, abstractmethod
from functools import wraps
from typing import Union, Set, List
from datetime import datetime

import gzip
import bz2

import json

from jax.numpy import exp, log
from .base_cov import Covariance
from .util import (
    make_serializable,
    deserialize,
    ensure_2d,
    make_multi_time_argument,
    object_str,
)
from .derivatives import (
    gradient,
    hessian,
    hessian_log_determinant,
)
from .validation import (
    _validate_time_x,
    _validate_float,
    _validate_array,
    _validate_bool,
)


logger = logging.getLogger("mellon")


class Predictor(ABC):
    """
    Abstract base class for predictor models. It provides a common interface for all subclasses, which are expected to
    implement the `_mean` method for making predictions.

    An instance `predictor` of a subclass of `Predictor` can be used to make a prediction by calling it with input data `x`:

    >>> y = predictor(x)

    It is the responsibility of subclasses to define the behaviour of `_mean`.

    Methods
    -------
    __call__(x: Union[array-like, pd.DataFrame], normalize: bool = False):

        Equivalent to calling the `mean` method, this uses the trained model to make
        predictions based on the input array, x.

        The prediction corresponds to the mean of the Gaussian Process conditional
        distribution of predictive functions.

        The input array must be 2D with the length of its second dimension matching the number
        of features used in training the model.

        Parameters
        ----------
        x : array-like
            The input data to the predictor, having shape (n_samples, n_input_features).
        normalize : bool
            Optional normalization by subtracting log(self.n_obs)
            (number of cells trained on), applicable only for cell-state density predictions.
            Default is False.

        Returns
        -------
        array
            The predicted output generated by the model.

        Raises
        ------
        ValueError
            If the number of features in 'x' does not align with the number of features the
            predictor was trained on.

    Attributes
    ----------
    n_obs : int
        The number of samples or cells that the model was trained on. This attribute is critical
        for normalization purposes, particularly when the `normalize` parameter in the `__call__`
        method is set to `True`.

    n_input_features : int
        The number of features/dimensions of the cell-state representation the predictor was
        trained on. This is used for validation of input data.
    """

    # number of features of input data (x.shape[1]) to be specified in __init__
    n_input_features: int

    # number of observations trained on (x.shape[0]) to be specified in __init__
    n_obs: int

    # a set of attribute names that should be saved to reconstruct the object
    _state_variables: Union[Set, List]

    @abstractmethod
    def __init__(self):
        """Initialize the predictor. Must be overridden by subclasses."""
        pass

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        n_obs = "None" if self.n_obs is None else f"{self.n_obs:,}"
        string = (
            'A predictor of class "'
            + self.__class__.__name__
            + '" with covariance function "'
            + repr(self.cov_func)
            + f'" trained on {n_obs} observations '
            + f"with {self.n_input_features:,} features "
            + "and data:\n"
            + "\n".join(
                [
                    str(key) + ": " + object_str(v)
                    for key, v in self._data_dict().items()
                ]
            )
        )
        return string

    @abstractmethod
    def _mean(self, *args, **kwargs):
        """Call the predictor. Must be overridden by subclasses."""

    def mean(self, x, normalize=False):
        """
        Use the trained model to make a prediction based on the input array, x.

        The prediction represents the mean of the Gaussian Process conditional
        distribution of predictive functions.

        The input array should be 2D with its second dimension's length
        equal to the number of features used in training the model.

        Parameters
        ----------
        x : array-like
            The input data to the predictor.
            The array should have shape (n_samples, n_input_features).
        normalize : bool
            Whether to normalize the value by subtracting log(self.n_obs)
            (number of cells trained on). Applicable only for cell-state density predictions.
            Default is False.

        Returns
        -------
        array
            The predicted output generated by the model.

        Raises
        ------
        ValueError
            If the number of features in 'x' does not match the
            number of features the predictor was trained on.
        """
        x = _validate_array(x, "x")
        x = ensure_2d(x)
        normalize = _validate_bool(normalize, "normalize")

        if x.shape[1] != self.n_input_features:
            raise ValueError(
                f"The predictor was trained on data with {self.n_input_features} features. "
                f"However, the provided input data has {x.shape[1]} features. "
                "Please ensure that the input data has the same number of features as the training data."
            )
        if normalize:
            if self.n_obs is None or self.n_obs == 0:
                message = (
                    "Cannot normalize without n_obs. Please set self.n_obs to the number "
                    "of samples/cells trained on to enable normalization."
                )
                logger.error(message)
                raise ValueError(message)
            logger.warn(
                'The normalization is only effective if the density was trained with d_method="fractal".'
            )
            return self._mean(x) - log(self.n_obs)
        else:
            return self._mean(x)

    __call__ = mean

    @abstractmethod
    def _covariance(self, *args, **kwars):
        """Compute the covariance. Must be overridden by subclasses."""

    def covariance(self, x, diag=True):
        """
        Computes the covariance of the Gaussian Process distribution of functions
        over new data points or cell states.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The new data points for which to compute the covariance.
        diag : boolean, optional (default=True)
            Whether to return the variance (True) or the full covariance matrix (False).

        Returns
        -------
        var : array-like, shape (n_samples,)
            If diag=True, returns the variances for each sample.
        cov : array-like, shape (n_samples, n_samples)
            If diag=False, returns the full covariance matrix between samples.
        """
        x = _validate_array(x, "x")
        x = ensure_2d(x)
        if x.shape[1] != self.n_input_features:
            raise ValueError(
                f"The predictor was trained on data with {self.n_input_features} features. "
                f"However, the provided input data has {x.shape[1]} features. "
                "Please ensure that the input data has the same number of features as the training data."
            )
        return self._covariance(x, diag=diag)

    @abstractmethod
    def _mean_covariance(self, *args, **kwars):
        """Compute the covariance of the mean. Must be overridden by subclasses."""

    def mean_covariance(self, x, diag=True):
        """
        Computes the uncertainty of the mean of the Gaussian process induced by
        the uncertainty of the latent representation of the mean function.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The new data points for which to compute the uncertainty.
        diag : boolean, optional (default=True)
            Whether to compute the variance (True) or the full covariance matrix (False).

        Returns
        -------
        var : array-like, shape (n_samples,)
            If diag=True, returns the variances for each sample.
        cov : array-like, shape (n_samples, n_samples)
            If diag=False, returns the full covariance matrix between samples.
        """
        x = _validate_array(x, "x")
        x = ensure_2d(x)
        if x.shape[1] != self.n_input_features:
            raise ValueError(
                f"The predictor was trained on data with {self.n_input_features} features. "
                f"However, the provided input data has {x.shape[1]} features. "
                "Please ensure that the input data has the same number of features as the training data."
            )
        return self._mean_covariance(x, diag=diag)

    def uncertainty(self, x, diag=True):
        """
        Computes the total uncertainty of the predicted values quantified by their variance
        or covariance.

        The total uncertainty is defined by `.covariance` + `.mean_covariance`.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The new data points for which to compute the uncertainty.
        diag : bool, optional (default=True)
            Whether to compute the variance (True) or the full covariance matrix (False).

        Returns
        -------
        var : array-like, shape (n_samples,) if diag=True
            The variances for each sample in the new data points.
        cov : array-like, shape (n_samples, n_samples) if diag=False
            The full covariance matrix between the samples in the new data points.
        """
        x = _validate_array(x, "x")
        x = ensure_2d(x)
        if x.shape[1] != self.n_input_features:
            raise ValueError(
                f"The predictor was trained on data with {self.n_input_features} features. "
                f"However, the provided input data has {x.shape[1]} features. "
                "Please ensure that the input data has the same number of features as the training data."
            )
        return self._covariance(x, diag=diag) + self._mean_covariance(x, diag=diag)

    def _data_dict(self):
        """Returns a dictionary containing the predictor's state data.
        All arrays nee to be (jax) numpy arrays for serialization.

        :return: A dictionary containing the predictor's state data.
        :rtype: dict
        """
        return {key: getattr(self, key) for key in self._state_variables}

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
        x = _validate_array(x, "x")
        x = ensure_2d(x)

        return gradient(self._mean, x, jit=jit)

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
        x = _validate_array(x, "x")
        x = ensure_2d(x)
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
        x = _validate_array(x, "x")
        x = ensure_2d(x)
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
        metamodule = import_module(module_name.split(".")[0])
        module = import_module(module_name)
        metaversion = getattr(metamodule, "__version__", "NA")
        version = getattr(module, "__version__", metaversion)
        data = self._data_dict()
        data.update(
            {
                "n_input_features": self.n_input_features,
                "n_obs": self.n_obs,
                "_state_variables": self._state_variables,
            }
        )
        data = {k: make_serializable(v) for k, v in data.items()}

        state = {
            "data": data,
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
        clsname = data_dict["metadata"]["classname"]
        module_name = data_dict["metadata"]["module_name"]
        module_version = data_dict["metadata"]["module_version"]

        if version.parse(module_version) < version.parse("1.4.0"):
            message = (
                f"Loading a predictor written by mellon {module_version} < 1.4.0. "
                "Please set predictor.n_obs to enable normalization."
            )
            logger.warning(message)
            if module_name == "mellon.conditional":
                clsname = clsname.replace("ConditionalMean", "Conditional")
            data_dict["data"]["n_obs"] = data_dict["data"].get("n_obs", None)
            state_vars = set(data_dict["data"].keys()) - {
                "n_input_features",
            }
            data_dict["data"]["_state_variables"] = data_dict["data"].get(
                "_state_variables", state_vars
            )

        module = import_module(module_name)
        Subclass = getattr(module, clsname)
        instance = Subclass.__new__(Subclass)
        instance.__setstate__(data_dict)

        return instance

    @classmethod
    def from_json_str(cls, json_str):
        """Deserialize the predictor from a JSON string.

        This method deserializes the predictor from the content of a JSON file.

        :param json_str: The JSON string from which to deserialize the predictor.
        :type json_str: str
        :return: An instance of the predictor.
        :rtype: Predictor subclass instance
        """
        data_dict = json.loads(json_str)
        return cls.from_dict(data_dict)


class ExpPredictor(Predictor):
    """
    Abstract base class for predictor models which returs the exponent of its `_mean` method upon a call.

    An instance `predictor` of a subclass of `Predictor` can be used to make a prediction by calling it with input data `x`:

    >>> y = predictor(x)

    It is the responsibility of subclasses to define the behaviour of `_mean`.
    """

    def mean(self, x, logscale=False):
        """
        Use the trained model to make a prediction based on the input array, x.

        The input array should be 2D with its second dimension's length
        equal to the number of features used in training the model.

        Parameters
        ----------
        x : array-like
            The input data to the predictor.
            The array should have shape (n_samples, n_input_features).

        logscale : bool
            Weather the predicted value should be returned in log scale.
            Default is False.

        Returns
        -------
        array
            The predicted output generated by the model.

        Raises
        ------
        ValueError
            If the number of features in 'x' does not match the
            number of features the predictor was trained on.
        """
        x = _validate_array(x, "x")
        logscale = _validate_bool(logscale, "logscale")
        x = ensure_2d(x)
        if x.shape[1] != self.n_input_features:
            raise ValueError(
                f"The predictor was trained on data with {self.n_input_features} features. "
                f"However, the provided input data has {x.shape[1]} features. "
                "Please ensure that the input data has the same number of features as the training data."
            )
        if logscale:
            return self._mean(x)
        return exp(self._mean(x))

    __call__ = mean

    @wraps(Predictor.covariance)
    def covariance(self, *args, **kwargs):
        logger.warning(
            "The covariance will be computed for the predicted value in log scale."
        )
        return super().covariance(*args, **kwargs)

    @wraps(Predictor.mean_covariance)
    def mean_covariance(self, *args, **kwargs):
        logger.warning(
            "The mean_covariance will be computed for the predicted value in log scale."
        )
        return super().mean_covariance(*args, **kwargs)

    @wraps(Predictor.uncertainty)
    def uncertainty(self, *args, **kwargs):
        logger.warning(
            "The uncertainty will be computed for the predicted value in log scale."
        )
        return super().uncertainty(*args, **kwargs)


class PredictorTime(Predictor):
    """
    Abstract base class for predictor models with a time covariate.

    An instance `predictor` of a subclass of `PredictorTime` can be used to
    make a prediction by calling it with input data `x` and `time`:

    >>> y = predictor(x, time)

    It is the responsibility of subclasses to define the behaviour of `_mean`.

    Methods
    -------
    __call__(x: Union[array-like, pd.DataFrame], normalize: bool = False):

        Equivalent to calling the `mean` method, this uses the trained model to make
        predictions based on the input array 'Xnew',
        considering the specified 'time' or 'multi_time'.

        The predictions represent the mean of the Gaussian Process conditional
        distribution of predictive functions.

        If 'time' is a scalar, it will be converted into a 1D array of the same size as 'Xnew'.

        Parameters
        ----------
        Xnew : array-like
            The new data points for prediction.
        time : scalar or array-like, optional
            The time points associated with each row in 'Xnew'.
            If 'time' is a scalar, it will be converted into a 1D array of the same size as 'Xnew'.
        normalize : bool
            Optional normalization by subtracting log(self.n_obs)
            (number of cells trained on), applicable only for cell-state density predictions.
            Default is False.

        Returns
        -------
        array
            The predicted output generated by the model.

        Raises
        ------
        ValueError
            If the number of features in 'x' does not align with the number of features the
            predictor was trained on.

    Attributes
    ----------
    n_obs : int
        The average number of samples or cells per time point that the model was trained on.
        This attribute is critical for normalization purposes, particularly when the `normalize`
        parameter in the `__call__` method is set to `True`.

    n_input_features : int
        The number of features/dimensions of the cell-state representation the predictor was
        trained on. This is used for validation of input data.
    """

    @make_multi_time_argument
    def mean(self, Xnew, time=None, normalize=False):
        """
        Use the trained model to make predictions based on the input array 'Xnew',
        considering the specified 'time' or 'multi_time'.

        The predictions represent the mean of the Gaussian Process conditional
        distribution of predictive functions.

        If 'time' is a scalar, it will be converted into a 1D array of the same size as 'Xnew'.

        Parameters
        ----------
        Xnew : array-like
            The new data points for prediction.
        time : scalar or array-like, optional
            The time points associated with each row in 'Xnew'.
            If 'time' is a scalar, it will be converted into a 1D array of the same size as 'Xnew'.
        normalize : bool
            Whether to normalize the value by subtracting log(self.n_obs)
            (number of cells trained on). Applicable only for cell-state density predictions.
            Default is False.
        multi_time : array-like, optional
            If 'multi_time' is specified then a prediction for all states in x will
            be made for each time value in multi_time separatly.

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
        Xnew = _validate_time_x(
            Xnew, time, n_features=self.n_input_features, cast_scalar=True
        )
        normalize = _validate_bool(normalize, "normalize")

        if normalize:
            if self.n_obs is None or self.n_obs == 0:
                message = (
                    "Cannot normalize without n_obs. Please set self.n_obs to the number "
                    "of samples/cells (per time point) trained on to enable normalization."
                )
                logger.error(message)
                raise ValueError(message)
            logger.warn(
                'The normalization is only effective if the density was trained with d_method="fractal".'
            )
            return self._mean(Xnew) - log(self.n_obs)
        else:
            return self._mean(Xnew)

    __call__ = mean

    @make_multi_time_argument
    def covariance(self, Xnew, time=None, diag=True):
        """
        Computes the covariance of the Gaussian Process distribution of functions
        over new data points or cell states.

        Parameters
        ----------
        Xnew : array-like, shape (n_samples, n_features)
            The new data points for which to compute the covariance.
        time : scalar or array-like, optional
            The time points associated with each cell/row in 'Xnew'.
            If 'time' is a scalar, it will be converted into a 1D array of the same size as 'Xnew'.
        diag : boolean, optional (default=True)
            Whether to return the variance (True) or the full covariance matrix (False).
        multi_time : array-like, optional
            If 'multi_time' is specified then a covariance for all states in x will
            be computed for each time value in multi_time separatly.

        Returns
        -------
        var : array-like, shape (n_samples,)
            If diag=True, returns the variances for each sample.
        cov : array-like, shape (n_samples, n_samples)
            If diag=False, returns the full covariance matrix between samples.
        """
        # if time is a scalar, convert it into a 1D array of the same size as Xnew
        Xnew = _validate_time_x(
            Xnew, time, n_features=self.n_input_features, cast_scalar=True
        )
        return self._covariance(Xnew, diag=diag)

    @make_multi_time_argument
    def mean_covariance(self, Xnew, time=None, diag=True):
        """
        Computes the uncertainty of the mean of the Gaussian process induced by
        the uncertainty of the latent representation of the mean function.

        Parameters
        ----------
        Xnew : array-like, shape (n_samples, n_features)
            The new data points for which to compute the uncertainty.
        time : scalar or array-like, optional
            The time points associated with each cell/row in 'Xnew'.
            If 'time' is a scalar, it will be converted into a 1D array of the same size as 'Xnew'.
        diag : boolean, optional (default=True)
            Whether to compute the variance (True) or the full covariance matrix (False).
        multi_time : array-like, optional
            If 'multi_time' is specified then a mean covariance for all states in x will
            be computed for each time value in multi_time separatly.

        Returns
        -------
        var : array-like, shape (n_samples,)
            If diag=True, returns the variances for each sample.
        cov : array-like, shape (n_samples, n_samples)
            If diag=False, returns the full covariance matrix between samples.
        """
        # if time is a scalar, convert it into a 1D array of the same size as Xnew
        Xnew = _validate_time_x(
            Xnew, time, n_features=self.n_input_features, cast_scalar=True
        )
        return self._mean_covariance(Xnew, diag=diag)

    @make_multi_time_argument
    def uncertainty(self, Xnew, time=None, diag=True):
        """
        Computes the total uncertainty of the predicted values quantified by their variance
        or covariance.

        The total uncertainty is defined by `.covariance` + `.mean_covariance`.

        Parameters
        ----------
        Xnew : array-like, shape (n_samples, n_features)
            The new data points for which to compute the uncertainty.
        time : scalar or array-like, optional
            The time points associated with each cell/row in 'Xnew'.
            If 'time' is a scalar, it will be converted into a 1D array of the same size as 'Xnew'.
        diag : bool, optional (default=True)
            Whether to compute the variance (True) or the full covariance matrix (False).
        multi_time : array-like, optional
            If 'multi_time' is specified then a uncertainty for all states in x will
            be computed for each time value in multi_time separatly.

        Returns
        -------
        var : array-like, shape (n_samples,) if diag=True
            The variances for each sample in the new data points.
        cov : array-like, shape (n_samples, n_samples) if diag=False
            The full covariance matrix between the samples in the new data points.
        """
        Xnew = _validate_time_x(
            Xnew, time, n_features=self.n_input_features, cast_scalar=True
        )
        return self._covariance(Xnew, diag=diag) + self._mean_covariance(
            Xnew, diag=diag
        )

    @make_multi_time_argument
    def time_derivative(
        self,
        x,
        time,
        jit=True,
    ):
        R"""
        Computes the time derivative of the prediction function for each line in `x`.

        This function applies a jax-based gradient operation to the density
        function evaluated at a specific time.
        The derivative is with respect to time and not the inputs in `x`.

        Parameters
        ----------
        x : array-like
            Data points where the derivative is to be evaluated.
        time : array-like or scalar
            Time point or points at which to evaluate the derivative.
            If `time` is a scalar, the derivative will be computed at this
            specific time point for all data points in `x`.
            If `time` is an array, it should be 1-D and the time derivative
            will be computed for all data-points at the corresponding time in the array.
        jit : bool, optional
            If True, use JAX's just-in-time (JIT) compilation to speed up the computation. Defaults to True.
        multi_time : array-like, optional
            If 'multi_time' is specified then a time derivative for all states in x will
            be computed for each time value in multi_time separatly.

        Returns
        -------
        array-like
            The time derivative of the prediction function evaluated at each point in `x`.
            The shape of the output array is the same as `x`.

        """
        Xnew = _validate_time_x(
            x, time, n_features=self.n_input_features, cast_scalar=True
        )
        return super().gradient(Xnew)[:, -1]

    @make_multi_time_argument
    def gradient(self, x, time, jit=True):
        """
        Computes the gradient of the prediction function for each point in `x` at a given time.

        Parameters
        ----------
        x : array-like
            Data points at which the gradient is to be computed.
        time : float
            Specific time point at which to compute the gradient.
        jit : bool, optional
            If True, use JAX's just-in-time (JIT) compilation to speed up the computation. Defaults to True.
        multi_time : array-like, optional
            If 'multi_time' is specified then a gradient for all states in x will
            be made for each time value in multi_time separatly.

        Returns
        -------
        array-like
            The gradient of the prediction function at each point in `x`.
            The shape of the output array is the same as `x`.
        """
        time = _validate_float(time, "time", optional=True)

        def dens_at(x):
            return self.mean(x, time)

        return gradient(dens_at, x, jit=jit)

    @make_multi_time_argument
    def hessian(self, x, time, jit=True):
        """
        Computes the Hessian matrix of the prediction function for each point in `x` at a given time.

        Parameters
        ----------
        x : array-like
            Data points at which the Hessian matrix is to be computed.
        time : float
            Specific time point at which to compute the Hessian matrix.
        multi_time : array-like, optional
            If 'multi_time' is specified then the computation will be made for each row.
        jit : bool, optional
            If True, use JAX's just-in-time (JIT) compilation to speed up the computation. Defaults to True.
        multi_time : array-like, optional
            If 'multi_time' is specified then a hessian for all states in x will
            be computed for each time value in multi_time separatly.

        Returns
        -------
        array-like
            The Hessian matrix of the prediction function at each point in `x`.
            The shape of the output array is `x.shape + x.shape[1:]`.
        """
        time = _validate_float(time, "time", optional=True)

        def dens_at(x):
            return self.__call__(x, time)

        return hessian(dens_at, x, jit=jit)

    @make_multi_time_argument
    def hessian_log_determinant(self, x, time, jit=True):
        """
        Computes the logarithm of the determinant of the Hessian of the prediction function
        for each point in `x` at a given time.

        Parameters
        ----------
        x : array-like
            Data points at which the log determinant is to be computed.
        time : float
            Specific time point at which to compute the log determinant.
        jit : bool, optional
            If True, use JAX's just-in-time (JIT) compilation to speed up the computation. Defaults to True.
        multi_time : array-like, optional
            If 'multi_time' is specified then a log determinant for all states in x will
            be computed for each time value in multi_time separatly.

        Returns
        -------
        array-like
            The sign of the determinant at each point in `x` and the logarithm of its absolute value.
            `signs.shape == log_determinants.shape == x.shape[0]`.
        """
        time = _validate_float(time, "time", optional=True)

        def dens_at(x):
            return self.__call__(x, time)

        return hessian_log_determinant(dens_at, x, jit=jit)
