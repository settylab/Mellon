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
from .version import __version__

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
    "setup_jax",
    "setup_logging",
]


def setup_jax(enable_x64=True, platform="cpu"):
    """Set up JAX configuration.

    Parameters
    ----------
    enable_x64 : bool, default=True
        Whether to enable 64-bit precision.
    platform : str, default="cpu"
        Platform to use for JAX computation.
    """
    jaxconfig.update("jax_enable_x64", enable_x64)
    jaxconfig.update("jax_platform_name", platform)


# Default JAX configuration
setup_jax()


# Logging configuration
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


def setup_logging(config=None):
    """Set up logging configuration.

    Parameters
    ----------
    config : dict, optional
        Logging configuration dictionary. If None, the default
        configuration is used.
    """
    if config is None:
        config = LOGGING_CONFIG
    logging.config.dictConfig(config)
    return logging.getLogger("mellon")


# Configure default logging
logger = setup_logging()
