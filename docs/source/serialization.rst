Serialization
=============

The Mellon module provides a comprehensive suite of tools for serializing and deserializing estimators. This functionality enables saving the state of an estimator, which can be used later for predictions without recomputing the estimator. It also makes it easy to share models across different systems, or to deploy a model in a production environment.

When an estimator is serialized, it includes a variety of metadata, including the serialization date, Python version, the class name of the estimator, and the Mellon version. This metadata serves as a record of the state of your environment at the time of serialization, which can be useful for reproducibility and debugging purposes.

All estimators in Mellon that inherit from the :class:`BaseEstimator` class, including :class:`mellon.model.DensityEstimator`, :class:`mellon.model.FunctionEstimator`, and :class:`mellon.model.DimensionalityEstimator`, have serialization capabilities.


Predictor Class
---------------

The `Predictor` class, accessible through the `predict` property of an estimator, handles the serialization and deserialization process.

.. autoclass:: mellon.Predictor
   :members:
   :undoc-members:
   :show-inheritance:

Serialization to AnnData
------------------------

Estimators can be serialized to an `AnnData`_ object. The `log_density` computation for the AnnData object shown below is a simplified example. For a more comprehensive guide on how to properly preprocess data to compute the log density, refer to our
`basic tutorial notebook <https://github.com/settylab/Mellon/blob/main/notebooks/basic_tutorial.ipynb>`_.


.. code-block:: python

    import mellon
    import anndata

    X = # your data here
    ad = anndata.AnnData(X=X)
    est = mellon.DensityEstimator()
    ad.obs["log_density"] = est.fit_predict(X)

    # Serialization
    ad.uns["log_density_funcyion"] = est.predict.to_dict()

    # Save the AnnData object
    ad.write('adata.h5ad')

Deserialization from AnnData
----------------------------

.. code-block:: python

    # Load the AnnData object
    ad = anndata.read('adata.h5ad')
    
    # Deserialization
    preddict = mellon.Predictor.from_dict(ad.uns["log_density_funcyion"])

    # User predictor on new data
    ad.obs["log_density"] = predict(X)

Serialization to File
---------------------

Mellon supports serialization to a human-readable JSON file and compressed file formats such as .gz (gzip) and .bz2 (bzip2). 

.. code-block:: python

    # Serialization to JSON
    est.predict.to_json("test_predictor.json")
    
    # Serialization with gzip compression
    est.predict.to_json("test_predictor.json.gz", compress="gzip")
    
    # Serialization with bzip2 compression
    est.predict.to_json("test_predictor.json.bz2", compress="bz2")

Deserialization from File
-------------------------

Mellon supports deserialization from JSON and compressed file formats. The compression method can be inferred from the file extension.

.. code-block:: python

    # Deserialization from JSON
    pred = mellon.Predictor.from_json("test_predictor.json")

    # Deserialization from compressed JSON
    pred = mellon.Predictor.from_json("test_predictor.json.gz")


.. _AnnData: https://anndata.readthedocs.io/en/latest/
