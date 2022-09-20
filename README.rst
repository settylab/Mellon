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
                        compute_landmarks, compute_nn, \
                        compute_mu, compute_ls, compute_L, \
                        compute_initial_value

   d = x.shape[1]
   rank = 0.999
   n_landmarks = 5000
   cov_func = Matern52
   jitter = 1e-6
   sigma2 = 1e-6

   landmarks = compute_landmarks(x, n_landmarks)
   nn_distances = compute_nn(x)
   mu = compute_mu(nn_distances, d)
   ls = compute_ls(nn_distances)
   cov_func = cov_func(ls)
   L = compute_L(x, cov_func, landmarks=landmarks, rank=rank, jitter=jitter)
   initial_value = compute_initial_value(nn_distances, d, mu, L)

   model = CrowdingEstimator(mu=mu, cov_func=cov_func,
                             ls=ls, nn_distances=nn_distances, initial_value=initial_value,
                             L=L, landmarks=landmarks, rank=rank, jitter=jitter, sigma2=sigma2)
   model = model.fit(x)
   train_log_density = model.predict(x)
   test_log_density = model.predict(y)

Covariance Functions:
======================

See the cov module for a list of covariance functions already implemented.
This section shows different ways to use a supplied covariance function
or define your own.

The cov_func argument supports a two argument function k(x, y) -> float.

.. code-block::
   :caption: Instantiate a predefined covariance function.

   from Crowding import Matern52
   cov_func = Matern52(ls)

.. code-block::
   :caption: Write a function of two variables.

   from Crowding import distance
   def Matern52_k(x, y):
       r = distance(x, y) / ls
       similarity = (sqrt(5.0) * r + square(sqrt(5.0) * r)/3 + 1) * exp(-sqrt(5.0) * r)
       return similarity
   cov_func = Matern52_k

The cov_func argument also supports a one argument function or class type
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
