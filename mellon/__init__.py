from jax.config import config as jaxconfig

jaxconfig.update("jax_enable_x64", True)
jaxconfig.update("jax_platform_name", "cpu")

from .base_cov import Covariance, Add, Mul, Pow
from .util import stabilize, mle, distance, DEFAULT_JITTER
from .decomposition import DEFAULT_RANK, DEFAULT_METHOD
from .cov import Matern32, Matern52, ExpQuad, Exponential, RatQuad
from .inference import (
    compute_transform,
    compute_loss_func,
    minimize_adam,
    minimize_lbfgsb,
    compute_log_density_x,
    compute_conditional_mean,
    DEFAULT_N_ITER,
    DEFAULT_INIT_LEARN_RATE,
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
    DEFAULT_N_LANDMARKS,
)
from .derivatives import (
    gradient,
    hessian,
    hessian_log_determinant,
)
from .model import (
    DensityEstimator,
    FunctionEstimator,
    DEFAULT_COV_FUNC,
)
