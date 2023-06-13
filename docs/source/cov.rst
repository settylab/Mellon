Covariance Functions
====================

In this section, we'll explore covariance functions, a key concept in Gaussian processes. The covariance function, or kernel, is a function of two inputs that gives some measure of similarity between them. In the context of our library, the `cov_func_curry` argument in the model class accepts a covariance class with a single argument (the length scale) for its `__init__` method, which should implement a function k(x,y) :math:`\rightarrow` float. If the length scale of the covariance function is not supplied as an argument, it will be calculated automatically. Alternatively, you can use the `cov_func` argument to pass an instance of a `Covariance` class.

Here is how you can pass a pre-existing covariance function class (which is the default behavior):

.. code-block:: python
   :caption: Pass a predefined covariance function class (Default behavior)

   from mellon.cov import Matern52
   cov_func_curry = Matern52
   cov_func = Matern52(length_scale)

If you want to write a custom covariance function k(x, y) :math:`\rightarrow` float, you can do so by inheriting from the `Covariance` base class. The `Covariance` base class's `__call__` method will call the function `k`.

.. code-block:: python
   :caption: Write a custom covariance function
    :math:`k(x, y) \rightarrow` float and inherit from the Covariance base class

   from mellon import distance
   from mellon import Covariance  # The Covariance base class __call__ method calls k.
   import jax.numpy as jnp

   class Matern52(Covariance):
       def __init__(self, ls=1.0):
           super().__init__()
           self.ls = ls

       def k(self, x, y):
           r = mellon.distance(x, y) / self.ls
           similarity = (
               jnp.sqrt(5.0) * r + jnp.square(jnp.sqrt(5.0) * r) / 3 + 1
           ) * jnp.exp(-jnp.sqrt(5.0) * r)
           return similarity

The `Covariance` class also supports arithmetic operations such as addition, multiplication, and exponentiation with the +, \*, and \*\* operators, respectively:

.. code-block:: python
   :caption: Combining two covariance functions.

   from mellon.cov import Matern52, ExpQuad
   cov_func = Matern52(length_scale)*.7 + ExpQuad(length_scale)*.3

Implemented Covariance Functions
--------------------------------

.. automodule:: mellon.cov
   :members:
   :undoc-members:
   :show-inheritance:

