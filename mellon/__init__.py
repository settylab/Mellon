from jax.config import config as jaxconfig

from .model import (
    DensityEstimator,
    FunctionEstimator,
    DimensionalityEstimator,
    TimeSensitiveDensityEstimator,
)
from .base_predictor import Predictor
from .cov import Covariance
from .util import Log

from . import _util as util
from . import _cov as cov
from . import _parameters as parameters
from . import _inference as inference
from . import _conditional as conditional
from . import _derivatives as derivatives

__version__ = "1.3.1"

__all__ = [
    "DensityEstimator",
    "FunctionEstimator",
    "DimensionalityEstimator",
    "TimeSensitiveDensityEstimator",
    "Predictor",
    "Covariance",
    "Log",
    "util",
    "cov",
    "model",
    "parameters",
    "inference",
    "conditional",
    "decomposition",
    "derivatives",
    "__version__",
]

# Set default configuration at import time
jaxconfig.update("jax_enable_x64", True)
jaxconfig.update("jax_platform_name", "cpu")
