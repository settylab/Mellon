import logging.config
import sys
from jax import config as jaxconfig

from .model import (
    DensityEstimator,
    FunctionEstimator,
    DimensionalityEstimator,
    TimeSensitiveDensityEstimator,
)
from .base_predictor import Predictor
from .cov import Covariance

from . import _util as util
from . import _cov as cov
from . import _parameters as parameters
from . import _inference as inference
from . import _conditional as conditional
from . import _derivatives as derivatives
from . import validation

__version__ = "1.6.0"

__all__ = [
    "DensityEstimator",
    "FunctionEstimator",
    "DimensionalityEstimator",
    "TimeSensitiveDensityEstimator",
    "Predictor",
    "Covariance",
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

# configure logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s] [%(levelname)-8s] %(message)s",
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": sys.stdout,
        },
    },
    "loggers": {
        "mellon": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("mellon")
