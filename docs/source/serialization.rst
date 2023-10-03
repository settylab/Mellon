.. _serialization:

Serialization
=============

The Mellon module facilitates the serialization and deserialization of
predictors, streamlining model transfer and reuse. This is especially relevant
in computational biology for applying pre-trained models to new datasets.
Serialization captures essential metadata like date, Python version, estimator
class, and Mellon version, ensuring research reproducibility and context for
the original training.

After fitting data, all Mellon models generate a predictor via their `.predict`
property, including model classes like :class:`mellon.model.DensityEstimator`,
:class:`mellon.model.TimeSensitiveDensityEstimator`,
:class:`mellon.model.FunctionEstimator`, and
:class:`mellon.model.DimensionalityEstimator`.


Predictor Class
---------------

All predictors inherit their serialization methods from :class:`mellon.Predictor`.

Serialization to AnnData
------------------------

Predictors can be serialized to an `AnnData`_ object. The `log_density`
computation for the AnnData object shown below is a simplified example. For a
more comprehensive guide on how to properly preprocess data to compute the log
density, refer to our
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

The function :meth:`mellon.Predictor.from_dict` can deserialize the
:class:`mellon.Predictor` and any sub class.

.. code-block:: python

    # Load the AnnData object
    ad = anndata.read('adata.h5ad')

    # Deserialization
    predictor = mellon.Predictor.from_dict(ad.uns["log_density_function"])

    # Use predictor on new data
    ad.obs["log_density"] = predictor(X)

Serialization to File
---------------------

Mellon supports serialization to a human-readable JSON file and compressed file
formats such as .gz (gzip) and .bz2 (bzip2).

The function :meth:`mellon.Predictor.from_json` can deserialize the
:class:`mellon.Predictor` and any sub class.

.. code-block:: python

    # Serialization to JSON
    est.predict.to_json("test_predictor.json")

    # Serialization with gzip compression
    est.predict.to_json("test_predictor.json.gz", compress="gzip")

    # Serialization with bzip2 compression
    est.predict.to_json("test_predictor.json.bz2", compress="bz2")

Deserialization from File
-------------------------

Mellon supports deserialization from JSON and compressed file formats. The
compression method can be inferred from the file extension.

.. code-block:: python

    # Deserialization from JSON
    predictor = mellon.Predictor.from_json("test_predictor.json")

    # Deserialization from compressed JSON
    predictor = mellon.Predictor.from_json("test_predictor.json.gz")


.. _AnnData: https://anndata.readthedocs.io/en/latest/
