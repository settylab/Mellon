from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

from .base_cov import Covariance, Add, Mul, Pow
from .util import stabilize, mle, distance, DEFAULT_JITTER
from .conditional import compute_conditional_mean, DEFAULT_SIGMA2
from .decomposition import compute_L, DEFAULT_RANK
from .cov import Matern32, Matern52, ExpQuad, Exponential, RatQuad
from .inference import compute_transform, compute_loss_func, run_inference, \
                       compute_pre_transformation, compute_loss, \
                       compute_log_density_x
from .parameters import compute_landmarks, compute_nn_distances, compute_d, compute_mu, \
                        compute_ls, compute_cov_func, compute_initial_value, \
                        DEFAULT_N_LANDMARKS
from .model import CrowdingEstimator, DEFAULT_COV_FUNC