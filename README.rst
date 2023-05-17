Mellon
======

.. image:: https://zenodo.org/badge/558998366.svg
   :target: https://zenodo.org/badge/latestdoi/558998366
.. image:: https://codecov.io/github/settylab/Mellon/branch/main/graph/badge.svg?token=TKIKXK4MPG 
    :target: https://app.codecov.io/github/settylab/Mellon
.. image:: https://www.codefactor.io/repository/github/settylab/mellon/badge/main
      :target: https://www.codefactor.io/repository/github/settylab/mellon/overview/main
      :alt: CodeFactor
.. image:: https://badge.fury.io/py/mellon.svg
       :target: https://badge.fury.io/py/mellon
.. image:: https://anaconda.org/conda-forge/mellon/badges/version.svg
       :target: https://anaconda.org/conda-forge/mellon
.. image:: https://static.pepy.tech/personalized-badge/mellon?period=total&units=international_system&left_color=grey&right_color=lightgrey&left_text=Downloads
    :target: https://pepy.tech/project/mellon

.. image:: https://github.com/settylab/mellon/raw/main/landscape.png?raw=true
   :target: https://github.com/settylab/Mellon

Mellon is a non-parametric cell-state density estimator based on a
nearest-neighbors-distance distribution. It uses a sparse gaussian process
to produce a differntiable density function that can be evaluated out of sample.

Installation
============

To install Mellon using **pip** you can run:

.. code-block:: bash

   pip install mellon

or to install using **conda** you can run:

.. code-block:: bash

   conda install -c conda-forge mellon

or to install using **mamba** you can run:

.. code-block:: bash

   mamba install -c conda-forge mellon

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

