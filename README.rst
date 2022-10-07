Crowding is a non-parametric density estimator based on the NearestNeighbors distribution.

Installation
============

To install with pip you can run:

.. code-block:: bash

   pip install Crowding

Basic Usage
===========

.. code-block:: python

    import Crowding as scd
    import numpy as np

    X = np.random.rand(100, 10)  # 10-dimensional state representation for 100 cells
    Y = np.random.rand(100, 10)  # arbitrary test data

    model = scd.CrowdingEstimator()
    log_density_x = model.fit_predict(X)
    log_density_y = model.predict(Y)


Usage with scanpy
=================

We recomend using a diffusion map latent representation of cell states as
input for the density computation. This latent represntation ensures that
euclidian distance relates to cell-state disimilarity, some meaningless noise
in cell-state is removed, and the dimensionality roghly represntes the
dimensionality of the phenotypic manifold.

.. code-block:: python

    import Crowding as scd
    import scanpy as sc

    adata = sc.read(h5ad_file_path)
    sc.external.tl.palantir(adata)
    
    model = scd.CrowdingEstimator()
    adata.obs['log_density'] = model.fit_predict(adata.obsm['DM_EigenVectors'])

Alternatively, to compute the density for a subset of cells on the complete
dataset, the density of the subset can be evaluated on all cells:

.. code-block:: python

    import Crowding as scd
    import scanpy as sc

    adata = sc.read(h5ad_file_path)
    sc.external.tl.palantir(adata)
    
    model = scd.CrowdingEstimator()
    mask = adata.obs['condition'] == 'subset_value' # arbitrary mask
    model.fit(adata[mask, :].obsm['DM_EigenVectors'])
    adata.obs['log_density_conditional'] = model.predict(adata.obsm['DM_EigenVectors'])

Parameters
==========

Above, all parameters are set automatically. The following code is equivalent.
Any parameters can be changed as desired.

.. code-block:: python

    import Crowding as scd
    import numpy as np

    X = np.random.rand(100, 10)  # 10-dimensional state representation for 100 cells


The Crowding density estimation uses the distance to the nearest neighbor
from each cell as the input data.

.. code-block:: python

    nn_distances = scd.compute_nn_distances(X)

One aspect of the density inference through Crowding is controlling 
the rate of density change between similar cells. This is realized
through a kernel function that computes the covariance of the log-density
values for pairs of cells. By default, we use the Matern52 kernel
with a heuristic for the length-scale parameter. This produces a twice
differentiable density function with reasonable rate of change. Variance,
bias, and differentiability can be controlled through the choice of kernel.
E.g., increasing the length-scale reduces variance and using `scd.ExpQuad`
increases differentiability.

.. code-block:: python

    length_scale = scd.compute_ls(nn_distances)
    cov_func = scd.Matern52(length_scale)


Landmarks in the data are used to approximate the covariance structure
and hence the similarity of density values between cells by using the similarity
to the landmarks as proxy. While any set of landmarks can be used, k-means-cluster
centroids preformed best in our tests. The number of landmarks limits the rank
of the resulting covariance matrix.

.. code-block:: python

    n_landmarks = 5000
    landmarks = scd.k_means(X, n_landmarks, n_init=1)[0]

By default, we further reduce the rank of the covariance matrix with an
improved Nyström approximation. The rank parameter can be used to either
select the fraction of *total variance* (sum of eigenvalues) preserved or
an integer number of ranks. The resulting `L` is a Cholesky factor of the
approximated covariance matrix.

.. code-block:: python

    rank = 0.999
    L = scd.compute_L(X, cov_func, landmarks=landmarks, rank=rank)


By default, we assume that the data can vary along all its dimensions.
However, if it is known that locally cells vary only along a
subspace, e.g., tangential to the phenotypic manifold, then the
dimensionality of this subspace should  be used.
`d` is used to correctly related the nearest-neighbor-distance
distribution to the cell-state density.

.. code-block:: python

    d = X.shape[1]

Crowding can automatically suggest a mean value `mu` for the Gaussian
process of log-density to ensure scale invariance. A low value ensures
that the density drops of quickly away from the data.

.. code-block:: python

    mu = scd.compute_mu(nn_distances, d)


An initial value, based on ridge regression, is used by default
to speed up the optimization.

.. code-block:: python

    initial_parameters = scd.compute_initial_value(nn_distances, d, mu, L)

    model = scd.CrowdingEstimator(
        n_landmarks=n_landmarks,
        rank=rank, method=method,
        jitter=jitter,
        landmarks=landmarks,
        nn_distances=nn_distances,
        d=d,
        mu=mu,
        ls=ls,
        cov_func=cov_func,
        L=L,
        initial_parameters=initial_parameters,
    )

    log_density_x = model.fit_predict(X)



Stages API
==========

Instead of fitting the model with the fit function, you may split training into
three stages: prepare_inference, run_inference, and process_inference.

.. code-block:: python

   model = scd.CrowdingEstimator()
   model.prepare_inference(X)
   model.run_inference()
   log_density_x = model.process_inference()

This allows you to make intermediate changes. For example, if you would
like to use your own optimizer, use the I/O of the three stages and
replace run_inference with your own optimizer:

.. code-block:: python

   def optimize(loss_func, initial_parameters):
       ...
       return optimal_parameters

   model = scd.CrowdingEstimator()
   loss_func, initial_parameters = model.prepare_inference(X)
   pre_transformation = optimize(loss_func, initial_parameters)
   log_density_x = model.process_inference(pre_transformation=pre_transformation)

Derivatives
===========

After inference the density and its derivatives can be computed for arbitrary
cell-states.

.. code-block:: python

    Y = np.random.rand(100, 10)  # arbitrary cell states

    log_density = model.predict(Y)
    gradients = model.gradient(Y)
    hessians = model.hessian(Y)

Of course this also works for `Y=X`.

