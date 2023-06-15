from jax.numpy import exp, unique, corrcoef, zeros, abs, stack
from jax.numpy import sum as arraysum
from jax.numpy.linalg import norm
from jaxopt import ScipyMinimize
from .density_estimator import DensityEstimator
from .util import Log

logger = Log()


def compute_ls_time(
    nn_distances,
    x,
    cov_func_curry,
    warn_below=500,
    return_data=False,
    density_estimator_kwargs=dict(),
):
    """
    Compute the optimal length scale for time (`ls_time`) based on density
    estimates and correlations at each unique time point.

    Parameters
    ----------
    nn_distances : array-like
        The nearest neighbor distances.
    x : array-like
        The training instances where the last column encodes the time point for each instance.
    cov_func_curry : function
        The covariance function curry.
    warn_below : int, optional
        The lower limit for the number of cells at a specific time point, below which a warning is issued.
    return_data : bool, optional
        Whether to return additional data along with `ls_time`. Defaults to False.
    density_estimator_kwargs : dict, optional
        Keyword arguments passed to the `DensityEstimator`. Defaults to an empty dict.

    Returns
    -------
    ls : float
        The optimal length scale for time (`ls_time`).
    """
    times = x[:, -1]
    states = x[:, :-1]
    unique_times = unique(times)
    n_times = len(unique_times)
    densities = []
    predictors = []

    for i, time in enumerate(unique_times):
        mask = times == time
        n_cells = arraysum(mask)
        logger.info(
            f"[{i+1} of {n_times}] Computing density for {n_cells:,} cells at time point {time}."
        )
        if n_cells < warn_below:
            logger.warning(
                f"Time point {time} only has {n_cells:,} cells. "
                "This could lead to inaccurate estimation of the time length scale `ls_time`."
            )

        x_at_time = x[mask, :-1]
        est = DensityEstimator(**density_estimator_kwargs)
        est.fit(x_at_time)
        densities.append(est.predict(states))
        predictors.append(est)

    densities = stack(densities)
    corrs = corrcoef(densities)
    delta_t = abs(unique_times.reshape(-1, 1) - unique_times.reshape(1, -1)).reshape(
        -1, 1
    )

    def ls_loss(log_ls):
        ls = exp(log_ls)
        covs = cov_func_curry(ls)(delta_t, zeros((1, 1))).reshape((n_times, n_times))
        return norm(covs - corrs)

    opt = ScipyMinimize(fun=ls_loss, method="L-BFGS-B", jit=False).run(0.0)
    ls = exp(opt.params)

    if return_data:
        return ls, densities, predictors

    return ls
