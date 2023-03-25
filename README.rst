.. image:: https://zenodo.org/badge/558998366.svg
   :target: https://zenodo.org/badge/latestdoi/558998366
.. image:: https://codecov.io/github/settylab/Mellon/branch/main/graph/badge.svg?token=TKIKXK4MPG 
    :target: https://codecov.io/github/settylab/Mellon

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
or use this
`basic tutorial notebook <https://github.com/settylab/Mellon/blob/main/notebooks/basic_tutorial.ipynb>`_.


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

