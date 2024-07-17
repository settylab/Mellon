Mellon
======

|zenodo| |codecov| |pypi| |conda|

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

Any of these calls should install Mellon and its dependencies within less than 1 minute.
If the dependency jax is not autimatically installed, please refer to https://github.com/google/jax.

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

Citations
=========

The Mellon manuscript is available on
`Nature Methods <https://www.nature.com/articles/s41592-024-02302-w>`_
and a preprint on
`bioRxiv <https://www.biorxiv.org/content/10.1101/2023.07.09.548272v1>`_.
If you use Mellon for your work, please cite our paper.

.. code-block:: bibtex

    @article{ottoQuantifyingCellstateDensities2024,
      title = {Quantifying Cell-State Densities in Single-Cell Phenotypic Landscapes Using {{Mellon}}},
      author = {Otto, Dominik J. and Jordan, Cailin and Dury, Brennan and Dien, Christine and Setty, Manu},
      date = {2024-06-18},
      journaltitle = {Nature Methods},
      issn = {1548-7105},
      doi = {10.1038/s41592-024-02302-w},
      url = {https://www.nature.com/articles/s41592-024-02302-w},
    }

You can find our
`reproducibility repository <https://github.com/settylab/MellonReproducibility>`_
to reproduce benchmarks and plots of the paper
`here <https://github.com/settylab/MellonReproducibility>`_.


.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8404223.svg
     :target: https://doi.org/10.5281/zenodo.8404223
.. |codecov| image:: https://codecov.io/github/settylab/Mellon/branch/main/graph/badge.svg?token=TKIKXK4MPG 
    :target: https://app.codecov.io/github/settylab/Mellon
.. |pypi| image:: https://badge.fury.io/py/mellon.svg
       :target: https://badge.fury.io/py/mellon
.. |conda| image:: https://anaconda.org/conda-forge/mellon/badges/version.svg
       :target: https://anaconda.org/conda-forge/mellon
