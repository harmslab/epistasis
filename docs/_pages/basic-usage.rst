Basic Usage
===========

All models in the ``epistasis`` package inherit scikit-learn classes and follow
a scikit-learn interface. If you are already familiar with the

Quick guide to epistasis models
-------------------------------

All ``epistasis`` models inherit a Scikit-learn regressor class and decorate each method
with new methods. (All the models
prepend ``Epistasis`` to the name of the scikit-learn object they inherit.) The easiest
way to calculate linear, epistasis in a genotype-phenotype map is to initialize
a model around a ``GenotypePhenotypeMap`` object (from the ``gpmap`` python package),
and call scikit-learn's ``fit`` method.

.. code-block:: python

    from gpmap import GenotypePhenotypeMap
    from epistasis.model import EpistasisLinearRegression

    # Create a genotype-phenotype map
    wildtype = "00"
    genotypes = ["00", "01", "10", "11"]
    phenotypes = [0, 1, 1, 4]
    gpm = GenotypePhenotypeMap("00", mutations)

    # Initialize an epistasis model for the genotype-phenotype map
    model = EpistasisLinearRegression.from_gpm(gpm)
    # Fit the epistasis model.
    model.fit()

The GenotypePhenotypeMap becomes an attribute of the model.

When ``fit`` is called on a model, the ``epistasis`` attribute is also exposed. This attribute is
an ``EpistasisMap`` object, which handles internal mapping for the epistatic coefficients and
includes a set of methods that make analyzing the epistatic coefficients easy.

To get a quick look at the epistatic coefficients:

    .. code-block:: python

        >>> model.epistasis.map("keys", "values")

            {
                "0": 0,
                "1": 1,
                "2": 1,
                "1,2": 2
            }

This object includes properties such as: ``keys``, ``values``, and ``labels``.
It also has a ``get_orders`` method, which returns submap of epistatic coefficients
with only the orders passed to it.
