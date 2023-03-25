from jax.config import config as jaxconfig

jaxconfig.update("jax_enable_x64", True)
jaxconfig.update("jax_platform_name", "cpu")

__version__ = "1.1.1"

from .base_cov import Covariance
from .util import stabilize, mle, distance, Log
from .cov import Matern32, Matern52, ExpQuad, Exponential, RatQuad
from .inference import (
    compute_transform,
    compute_loss_func,
    minimize_adam,
    minimize_lbfgsb,
    compute_log_density_x,
    compute_conditional_mean,
)
from .parameters import (
    compute_landmarks,
    k_means,
    compute_nn_distances,
    compute_d,
    compute_mu,
    compute_ls,
    compute_cov_func,
    compute_L,
    compute_initial_value,
)
from .derivatives import (
    gradient,
    hessian,
    hessian_log_determinant,
)
from .model import (
    DensityEstimator,
    FunctionEstimator,
)
