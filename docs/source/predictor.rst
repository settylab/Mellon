Predictors
==========

Predictors in the Mellon framework can be invoked directly via their `__call__`
method to produce function estimates at new locations. These predictors can
also double as Gaussian Processes, offering uncertainty estimattion options.
It also comes with serialization capabilities detailed in :ref:`serialization <serialization>`.

Basic Usage
-----------

To generate estimates for new, out-of-sample locations, instantiate a
predictor and call it like a function:

.. code-block:: python
   :caption: Example of accessing the :class:`mellon.Predictor` from the
		:class:`mellon.model.DensityEstimator` in Mellon Framework
   :name: example-usage-density-predictor

    model = mellon.model.DensityEstimator(...)  # Initialize the model with appropriate arguments
    model.fit(X)  # Fit the model to the data
    predictor = model.predict  # Obtain the predictor object
    predicted_values = predictor(Xnew)  # Generate predictions for new locations


Uncertainy
------------

If the predictor was generated with
uncertainty estimates (typically by passing `predictor_with_uncertainty=True`
and `optimizer="advi"` to the model class, e.g., :class:`mellon.model.DensityEstimator`)
then it provides methods for computing variance at these locations, and co-variance to any other
location.

- Variance Methods:
    - :meth:`mellon.Predictor.covariance`
    - :meth:`mellon.Predictor.mean_covariance`
    - :meth:`mellon.Predictor.uncertainty`

Sub-Classes
-----------

The `Predictor` module in the Mellon framework features a variety of
specialized subclasses of :class:`mellon.Predictor`. The specific subclass
instantiated by the model is contingent upon two key parameters:

- `gp_type`: This argument determines the type of Gaussian Process used internally.
- The nature of the predicted output: This can be real-valued, strictly positive, or time-sensitive.

The `gp_type` argument mainly affects the internal mathematical operations,
whereas the nature of the predicted value dictates the subclass's functional
capabilities:

- **Real-valued Predictions**: Such as log-density estimates, :class:`mellon.Predictor`.
- **Positive-valued Predictions**: Such as dimensionality estimates, :class:`mellon.base_predictor.ExpPredictor`.
- **Time-sensitive Predictions**: Such as time-sensitive density estimates :class:`mellon.base_predictor.PredictorTime`.



Vanilla Predictor
-----------------

Utilized in the following methods:

- :attr:`mellon.model.DensityEstimator.predict`
- :attr:`mellon.model.DimensionalityEstimator.predict_density`
- :attr:`mellon.model.FunctionEstimator.predict`

.. autoclass:: mellon.Predictor
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: n_obs, n_input_features

Exponential Predictor
---------------------

- Used in :attr:`mellon.model.DimensionalityEstimator.predict`
- Predicted values are strictly positive. Variance is expressed in log scale.

.. autoclass:: mellon.base_predictor.ExpPredictor
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: n_obs, n_input_features

Time-sensitive Predictor
------------------------

- Utilized in :attr:`mellon.model.TimeSensitiveDensityEstimator.predict`
- Special arguments `time` and `multi_time` permit time-specific predictions.

.. autoclass:: mellon.base_predictor.PredictorTime
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: n_obs, n_input_features

