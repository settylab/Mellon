Covariance Functions
====================

This section shows different ways to use a supplied covariance function
or define your own.

The `cov_func_curry` argument of the model class supports a one argument function
or class type that returns a function k(x, y)
:math:`\rightarrow` float. In this case, the length scale of the covariance
function will be computed automatically if not passed as an argument.
Alternatively, one can set the `cov_func` argument with a function taking
two arrays of cell states returning the similatity between each pair.

.. code-block:: python
   :caption: Pass a predefined covariance function class (Default behavior)

   from Crowding import Matern52
   cov_func_cury = Matern52
   cov_func = Matern52(length_scale)

.. code-block:: python
   :caption: Write a function of one variable that returns a function k(x, y) :math:`\rightarrow` float

   from Crowding import distance    # distance computes the distance between each point in x
                                    # and each point in y.
   def Matern52(ls=1.0):
       def k(x, y):
           r = distance(x, y) / ls
           similarity = (sqrt(5.0) * r + square(sqrt(5.0) * r)/3 + 1) * exp(-sqrt(5.0) * r)
           return similarity
       return k

.. code-block:: python
   :caption: Write a function of one variable that returns a function
    :math:`k(x, y) \rightarrow` float and inherit from the Covariance base class

   from Crowding import distance
   from Crowding import Covariance  # The Covariance base class __call__ method calls k.

   class Matern52(Covariance):
       def __init__(self, ls=1.0):
           super().__init__()
           self.ls = ls

       def k(self, x, y):
           r = distance(x, y) / self.ls
           similarity = (sqrt(5.0) * r + square(sqrt(5.0) * r)/3 + 1) * exp(-sqrt(5.0) * r)
           return simiAlarity

The Covariance class upports adding, multiplying, and taking the covariance
to a power with the +, \*, and \*\* operators:

.. code-block:: python
   :caption: Combining two covariance functions.

   from Crowding import Matern52, ExpQuad
   cov_func = Matern52(length_scale)*.7 + ExpQuad(length_scale)*.3

Implemented Covariance Functions
--------------------------------

.. automodule:: Crowding.cov
   :members:
   :undoc-members:
   :show-inheritance:
