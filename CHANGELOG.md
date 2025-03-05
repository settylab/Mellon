# 1.6.1

 - forward randome state to k-means to compute landmarks

# v1.6.0

 - use [PyNNDescent](https://github.com/lmcinnes/pynndescent) for faster nearest neighbor distance computation
 - update deprecated `tol` argument of `jax.numpy.linalg.matrix_rank` to `rtol`

# v1.5.0

 - remove `numpy` as direct dependency
 - bugfix DimensionalityEstimator dimensionality initialization
 - implement 'fixed' gaussian proces type to allow more inducing points than datapoints
 - implement `copy()` method for `Predictor` class
 - html representation for major objects
 - `covariance` of sparse `FunctionEstimator`

# v1.4.3

 - Detailed logging about invalid `nn_distances`.
 - Validation of `nn_distances` passed at initialization
 - Validate that passed scalars are not `nan`.
 - Rename all `_validate` methods to `validate`

# v1.4.2

 - Implement gradients for the covariance kernels through the `k_grad` method
 - Implement `mellon.cov.Linear` covariance kernel
 - Change logging setup to configuration dict
 - allow setting `active_dims` for composit kernels, allowing more flexible covariance kernel specifications
 - update jaxconfig impot for compatibility with newer jax versions
 - generalize variing sigma in FunctionEstimator for higher dimensional functions

# v1.4.1

Drop constraint on NumPy version `numpy<1.25.0` which was introdcuded due to
an incompatibility of `numpy==1.25.0` and `jax<0.4.16`.
See [Jax Change Log](https://jax.readthedocs.io/en/latest/changelog.html#jax-0-4-16-sept-18-2023).

# v1.4.0
## New Features
### `with_uncertainty` Parameter
Integrates a boolean parameter `with_uncertainty` across all estimators: [DensityEstimator](https://mellon.readthedocs.io/en/uncertainty/model.html#mellon.model.DensityEstimator), TimeSensitiveDensityEstimator, FunctionEstimator, and DimensionalityEstimator. It modifies the fitted predictor, accessible via the `.predict` property, to include the following methods:
 - `.covariance(X)`: Calculates the (co-)variance of the posterior Gaussian Process (GP).
   - Almost 0 near landmarks; grows for out-of-sample locations.
   - Increases with sparsity.
   - Defaults to `diag=True`, computing only the covariance matrix diagonal.
 - `.mean_covariance(X)`: Computes the (co-)variance through the uncertainty of the mean function's GP posterior.
   - Derived from Bayesian inference for latent density function representation.
   - Increases in low data or low-density areas.
   - Only available with posterior uncertainty quantification, e.g., `optimizer='advi'` except for the `FunctionEstimator` where input uncertainty is specified through the `sigma` parameter.
   - Defaults to `diag=True`, computing only the covariance matrix diagonal.
 - `.uncertainty(X)`: Combines `.covariance(X)` and `.mean_covariance(X)`.
   - Defaults to `diag=True`, computing only the covariance matrix diagonal.
   - Square root provides standard deviation.

### `gp_type` Parameter
Introduces the `gp_type` parameter to all relevant [estimators](https://mellon.readthedocs.io/en/uncertainty/model.html) to explicitly specify the Gaussian Process (GP) sparsification strategy, replacing the previously used `method` argument (with options auto, fixed, and percent) that implicitly controlled sparsification. The available options for `gp_type` include:
 - 'full': Non-sparse GP.
 - 'full_nystroem': Sparse GP with Nyström rank reduction, lowering computational complexity.
 - 'sparse_cholesky': SParse GP using landmarks/inducing points.
 - 'sparse_nystroem': Improved Nyström rank reduction on sparse GP with landmarks, balancing accuracy and efficiency.

This new parameter adds additional validation steps, ensuring that no contradictory parameters are specified. If inconsistencies are detected, a helpful reply guides the user on how to fix the issue. The value can be either a string matching one of the options above or an instance of the `mellon.parameters.GaussianProcessType` Enum. Partial matches log a warning, using the closest match. Defaults to 'sparse_cholesky'.

*Note: Nyström strategies are not applicable to the **FunctionEstimator**.*

### `y_is_mean` Parameter
Adds a boolean parameter `y_is_mean` to [FunctionEstimator](https://mellon.readthedocs.io/en/uncertainty/model.html#mellon.model.FunctionEstimator), affecting how `y` values are interpreted:
- **Old Behavior**: `sigma` impacted conditional mean functions and predictions.
- **Intermediate Behavior**: `sigma` only influenced prediction uncertainty.
- **New Parameter**: If `y_is_mean=True`, `y` values are treated as a fixed mean; `sigma` reflects only uncertainty. If `y_is_mean=False`, `y` is considered a noisy measurement, potentially smoothing values at locations `x`.

This change benefits DensityEstimator, TimeSensitiveDensityEstimator, and DimensionalityEstimator where function values are predicted for out-of-sample locations after mean GP computation.

### `check_rank` Parameter
Introduces the `check_rank ` parameter to all relevant [estimators](https://mellon.readthedocs.io/en/uncertainty/model.html). This boolean parameter explicitly controls whether the rank check is performed, specifically in the `gp_type="sparse_cholesky"` case. The rank check assesses the chosen landmarks for adequate complexity by examining the approximate rank of the covariance matrix, issuing a warning if insufficient. Allowed values are:
 - `True`: Always perform the check.
 - `False`: Never perform the check.
 - `None` (Default): Perform the check only if `n_landmarks` is greater than or equal to `n_samples` divided by 10.

The default setting aims to bypass unnecessary computation when the number of landmarks is so abundant that insufficient complexity becomes improbable.

### `normalize` Parameter

The `normalize` parameter is applicable to both the [`.mean`](https://mellon.readthedocs.io/en/uncertainty/serialization.html#mellon.Predictor.mean) method and `.__call__` method within the [mellon.Predictor](https://mellon.readthedocs.io/en/uncertainty/serialization.html#predictor-class) class. When set to `True`, these methods will subtract `log(number of observations)` from the value returned. This feature is particularly useful with the [DensityEstimator](https://mellon.readthedocs.io/en/uncertainty/model.html#mellon.model.DensityEstimator), where normalization adjusts for the number of cells in the training sample, allowing for accurate density comparisons between datasets. This correction takes into account the effect of dataset size, ensuring that differences in total cell numbers are not unduly influential. By default, the parameter is set to `False`, meaning that density differences due to variations in total cell number will remain uncorrected.

### `normalize_per_time_point` Parameter

This parameter fine-tunes the `TimeSensitiveDensityEstimator` to handle variations in sampling bias across different time points, ensuring both continuity and differentiability in the resulting density estimation. Notably, it also allows to reflect the growth of a population even if the same number of cells were sampled from each time point.

The normalization is realized by manipulating the nearest neighbor distances
`nn_distances` to reflect the deviation from an expected cell count.

- **Type**: Optional, accepts `bool`, `list`, `array-like`, or `dict`.

#### Options:

- **`True`:** Normalizes to emulate an even distribution of total cell count across all time points.
- **`False`:** Retains raw cell counts at each time point for density estimation.
- **List/Array-like**: Specifies an ordered sequence of total cell count targets for each time point, starting with the earliest.
- **Dict**: Associates each unique time point with a specific total cell count target.

#### Notes:

- **Relative Metrics**: While this parameter adjusts for sample bias, it only requires relative cell counts for comparisons within the dataset; exact counts are not mandatory.
- **`nn_distance` Precedence**: If `nn_distance` is supplied, this parameter will be bypassed, and the provided distances will be used directly.
- The default value is `False`


## Enhancements
 - Optimization by saving the intermediate result `Lp` in the estimators for reuse, enhancing the speed of the predictive function computation in non-Nyström strategies.
 - The `DimensionalityEstimator.predict` now returns a subclass of the `mellon.Predictor` class instead of a closure. Giving access to serialization and uncertainty computations.
 - Expanded testing.
 - propagate logging messages and explicit logger name "mellon" everywhere
 - extended parameter validation for the estimators now also applies to the `compute_L` function
 - better string representation of estimators and predictors
 - bugfix some edge cases
 - Revise some documentation (s. b70bb04a4e921ceab63b60026b8033e384a8916a) and include [Predictor](https://mellon.readthedocs.io/en/uncertainty/predictor.html) page on sphinx doc

## Changes

 - The mellon.Predictor class now has a method `.mean` that is an alias to `.__call__`.
 - All mellon.Predictor sub classes `...ConditionalMean...` were renamed to `...Conditional...` since they now also compute `.covariance` and `.mean_covariance`.
 - All generating methods for mellon.Predictor were renamed from `...conditional_mean...` to `conditional`.
 - A new log message now informs that the normalization is not effective `d_method != "fractal"`. Additionally, using `normalize=True` in the density predictor triggers a warning that one has to use the non default `d_method = "fractal"` in the `DensityEstimator`.
