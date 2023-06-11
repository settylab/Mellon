from jax.config import config as jaxconfig

jaxconfig.update("jax_enable_x64", True)
jaxconfig.update("jax_platform_name", "cpu")

__version__ = "1.2.0"

from .model import (
    DensityEstimator,
    FunctionEstimator,
    DimensionalityEstimator,
)
from .base_predictor import Predictor
from .cov import Covariance
from .util import Log

from . import cov
from . import model
from . import parameters
from . import inference
from . import conditional
from . import decomposition
from . import derivatives
