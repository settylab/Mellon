from jax.config import config as jaxconfig

from .model import (
    DensityEstimator,
    FunctionEstimator,
    DimensionalityEstimator,
)
from .base_predictor import Predictor
from .cov import Covariance
from .util import Log

from . import util
from . import cov
from . import model
from . import parameters
from . import inference
from . import conditional
from . import decomposition
from . import derivatives

__version__ = "1.3.0rc"

__all__ = [
    "DensityEstimator",
    "FunctionEstimator",
    "DimensionalityEstimator",
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
]

# Set default configuration at import time
jaxconfig.update("jax_enable_x64", True)
jaxconfig.update("jax_platform_name", "cpu")
