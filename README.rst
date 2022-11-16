Mellon is a non-parametric density estimator based on the NearestNeighbors distribution.

Installation
============

To install with pip you can run:

.. code-block:: bash

   pip install mellon

Documentation
=============

Please read the
`documentation <https://mellon.readthedocs.io/en/latest/index.html>`_
and this
`basic tutorial <https://mellon.readthedocs.io/en/latest/notebooks/basic_tutorial.html>`_.


Basic Usage
===========

.. code-block:: python

    import mellon
    import numpy as np

    X = np.random.rand(100, 10)  # 10-dimensional state representation for 100 cells
    Y = np.random.rand(100, 10)  # arbitrary test data

    model = mellon.DensityEstimator()
    log_density_x = model.fit_predict(X)
    log_density_y = model.predict(Y)

