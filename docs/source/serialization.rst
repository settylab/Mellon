Serialization
=============

The Mellon module provides a comprehensive suite of tools for serializing and deserializing estimators. This functionality is particularly useful in computational biology where models might be trained on one dataset and then used for making predictions on new datasets. The ability to serialize an estimator allows you to save the state of a model after it has been trained, and then load it later for predictions without needing to retrain the model.

For instance, you might have a large dataset on which you train an estimator. Training might take a long time due to the size and complexity of the data. With serialization, you can save the trained estimator and then easily share it with collaborators, apply it to new data, or use it in a follow-up study.

When an estimator is serialized, it includes a variety of metadata, including the serialization date, Python version, the class name of the estimator, and the Mellon version.
This metadata serves as a detailed record of the state of your environment at the time of serialization, which can be crucial for reproducibility in scientific research and for understanding the conditions under which the estimator was originally trained.

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
    est.fit(X)
    ad.obs["log_density"] = est.predict(X)

    # Serialization
    ad.uns["log_density_function"] = est.predict.to_dict()

    # Save the AnnData object
    ad.write('adata.h5ad')

Deserialization from AnnData
----------------------------

.. code-block:: python

    # Load the AnnData object
    ad = anndata.read('adata.h5ad')

    # Deserialization
    predictor = mellon.Predictor.from_dict(ad.uns["log_density_function"])

    # Use predictor on new data
    ad.obs["log_density"] = predictor(X)

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
    predictor = mellon.Predictor.from_json("test_predictor.json")

    # Deserialization from compressed JSON
    predictor = mellon.Predictor.from_json("test_predictor.json.gz")


.. _AnnData: https://anndata.readthedocs.io/en/latest/
