Crowding is a non-parametric density estimator based on the NearestNeighbors distribution.

Installation:
===============

To install with pip you can run:

.. code-block::

   pip install Crowding

Basic Usage:
======================

.. code-block::

   import numpy as np
   from Crowding import CrowdingEstimator

   x = np.random.rand(100, 10)  # arbitrary training data
   y = np.random.rand(100, 10)  # arbitrary test data

   model = CrowdingEstimator()
   model = model.fit(x)
   train_log_density = model.predict(x)
   test_log_density = model.predict(y)

Parameters:
======================

Above, all parameters are set automatically. The following code is equivalent.
Any parameters can be changed as desired.

.. code-block::

   from Crowding import CrowdingEstimator, Matern52, \
                        compute_landmarks, compute_nn_distances, compute_d, compute_mu, \
                        compute_ls, compute_cov_func, compute_initial_value


   cov_func_curry = Matern52

   # Higher n_landmarks is always better, but is more expensive.
   n_landmarks = 5000

   # Higher rank is always better, but is more expensive.
   rank = 0.999

   # Generally unnecessary, but it clarifies the ambiguous case where rank = 1 or 1.0.
   method = 'auto'

   # Increase if the covariance matrix is not positive definite.
   jitter = 1e-6

   sigma2 = 1e-6

   # Landmark points should summarize the training data. The best way to choose
   # landmark points is an open problem.
   landmarks = compute_landmarks(x, n_landmarks=n_landmarks)

   nn_distances = compute_nn_distances(x)

   # The data may vary in fewer dimensions at a local scale.
   d = compute_d(x)

   # mu should be smaller than the density at most datapoints so the
   # log density function decays to mu away from the data. You can use the mle
   # function from the util module as an initial noisy log density estimation.
   mu = compute_mu(nn_distances, d)

   # Higher length scale produces a smoother function.
   ls = compute_ls(nn_distances)

   cov_func = compute_cov_func(cov_func_curry, ls)

   L = compute_L(x, cov_func, landmarks=landmarks, rank=rank, method=method, jitter=jitter)

   # The initial_value should be in the same basin as the global minimum.
   initial_value = compute_initial_value(nn_distances, d, mu, L)

   model = CrowdingEstimator(cov_func_curry=cov_func_curry, n_landmarks=n_landmarks, \
                             rank=rank, method=method, jitter=jitter, sigma2=sigma2, \
                             landmarks=landmarks, nn_distances=nn_distances, d=d, \
                             mu=mu, ls=ls, cov_func=cov_func, L=L, \
                             initial_value=initial_value)
   model = model.fit(x)
   train_log_density = model.predict(x)
   test_log_density = model.predict(y)

Updating Parameters:
======================

After fitting the model, you may want to change some parameters without recomputing
every step. For example, suppose you want to increase mu by 5.

.. code-block::

   model = model.fit(x)

   model.recursive_setattr('mu', model.mu + 5)
   model = model.fit(x)

recursive_setattr will update the attribute to the given value and set any attributes
of model that depend on the updated attribute to None. This means that when the model
is fit again, only the attributes that depend on the updated attribute will be recomputed.

However, recursive_setattr will not set attributes to None if their value was set
explicitly, either as a parameter or through assignment. For example, initial_value
depends on mu, but the call recursive_setattr('mu', model.mu + 5) will not affect
initial_value in any of the following:

.. code-block::

   initial_value = np.random.normal(0, 1, size=100)  # arbitrary initial_value
   model = Crowding(initial_value=initial_value)
   model = model.fit(x)

   model.recursive_setattr('mu', model.mu + 5)
   model = model.fit(x)

.. code-block::

   model = Crowding()
   model = model.fit(x)

   initial_value = np.random.normal(0, 1, size=100)  # arbitrary initial_value
   model.recursive_setattr('initial_value', initial_value)  # Will update initial_value
   model = model.fit(x)

   model.recursive_setattr('mu', model.mu + 5)  # Will update mu, but not initial_value
   model = model.fit(x)

Attribute Dependencies:
------------------------

When an attribute is explicitly updated, here are the attributes that will be reset if they were implicitly computed:

x : [landmarks, nn_distances, d, mu, ls, cov_func, L, initial_value, transform, loss_func, optimize_result, pre_transformation, log_density_x, log_density_func]

cov_func_curry : [cov_func, L, initial_value, transform, loss_func, optimize_result, pre_transformation, log_density_x, log_density_func]

n_landmarks : [landmarks, L, initial_value, transform, loss_func, optimize_result, pre_transformation, log_density_x, log_density_func]

rank : [L, initial_value, transform, loss_func, optimize_result, pre_transformation, log_density_x, log_density_func]

method : [L, initial_value, transform, loss_func, optimize_result, pre_transformation, log_density_x, log_density_func]

jitter : [L, initial_value, transform, loss_func, optimize_result, pre_transformation, log_density_x, log_density_func]

sigma2 : [log_density_func]

n_iter : [optimize_result, pre_transformation, log_density_x, log_density_func]

init_learn_rate : [optimize_result, pre_transformation, log_density_x, log_density_func]

inference_func : [optimize_result, pre_transformation, log_density_x, log_density_func]

landmarks : [L, initial_value, transform, loss_func, optimize_result, pre_transformation, log_density_x, log_density_func]

nn_distances : [mu, ls, cov_func, L, initial_value, transform, loss_func, optimize_result, pre_transformation, log_density_x, log_density_func]

d : [mu, initial_value, transform, loss_func, optimize_result, pre_transformation, log_density_x, log_density_func]

mu : [initial_value, transform, loss_func, optimize_result, pre_transformation, log_density_x, log_density_func]

ls : [cov_func, L, initial_value, transform, loss_func, optimize_result, pre_transformation, log_density_x, log_density_func]

cov_func : [L, initial_value, transform, loss_func, optimize_result, pre_transformation, log_density_x, log_density_func]

L : [initial_value, transform, loss_func, optimize_result, pre_transformation, log_density_x, log_density_func]

initial_value : [optimize_result, pre_transformation, log_density_x, log_density_func]

transform : [loss_func, optimize_result, pre_transformation, log_density_x, log_density_func]

loss_func : [optimize_result, pre_transformation, log_density_x, log_density_func]

optimize_result : []

pre_transformation : [log_density_x, log_density_func]

log_density_x : [log_density_func]

log_density_func : []

Covariance Functions:
======================

See the cov module for a list of covariance functions already implemented.
This section shows different ways to use a supplied covariance function
or define your own.

The cov_func_curry argument supports a one argument function or class type
that returns a function k(x, y) -> float. In this case, the length scale
of the covariance function will be set to ls, which is computed automatically
if not passed as an argument.

.. code-block::
   :caption: Pass a predefined covariance function class (Default behavior)

   from Crowding import Matern52
   cov_func = Matern52

.. code-block::
   :caption: Write a function of one variable that returns a function k(x, y) -> float

   from Crowding import distance    # distance computes the distance between each point in x
                                    # and each point in y.
   def Matern52(ls=1.0):
       def k(x, y):
           r = distance(x, y) / ls
           similarity = (sqrt(5.0) * r + square(sqrt(5.0) * r)/3 + 1) * exp(-sqrt(5.0) * r)
           return similarity
       return cov_func
   cov_func = Matern52

.. code-block::
   :caption: Inherit from the Covariance base class

   from Crowding import distance
   from Crowding import Covariance  # The Covariance base class __call__ method calls k.
                                    # It also supports adding, multiplying, and exponentiating
                                    # with the +, *, and ** operators.

   class Matern52(Covariance):
       def __init__(self, ls=1.0):
           super().__init__()
           self.ls = ls

       def k(self, x, y):
           r = distance(x, y) / self.ls
           similarity = (sqrt(5.0) * r + square(sqrt(5.0) * r)/3 + 1) * exp(-sqrt(5.0) * r)
           return similarity
   cov_func = Matern52

Alternatively, the cov_func argument supports a two argument function k(x, y) -> float.

.. code-block::
   :caption: Instantiate a predefined covariance function.

   from Crowding import Matern52

   ls = 1.0  # Set ls as desired.
   cov_func = Matern52(ls)

.. code-block::
   :caption: Write a function of two variables.

   from Crowding import distance

   ls = 1.0  # Set ls as desired.
   def Matern52_k(x, y):
       r = distance(x, y) / ls
       similarity = (sqrt(5.0) * r + square(sqrt(5.0) * r)/3 + 1) * exp(-sqrt(5.0) * r)
       return similarity
   cov_func = Matern52_k

.. code-block::
   :caption: Instatiate a type that inherits from the Covariance base class.

   from Crowding import distance
   from Crowding import Covariance  # The Covariance base class __call__ method calls k.
                                    # It also supports adding, multiplying, and exponentiating
                                    # with the +, *, and ** operators.

   class Matern52(Covariance):
       def __init__(self, ls=1.0):
           super().__init__()
           self.ls = ls

       def k(self, x, y):
           r = distance(x, y) / self.ls
           similarity = (sqrt(5.0) * r + square(sqrt(5.0) * r)/3 + 1) * exp(-sqrt(5.0) * r)
           return similarity

   ls = 1.0  # Set ls as desired.
   cov_func = Matern52(ls)