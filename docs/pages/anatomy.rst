Anatomy of an epistasis model
=============================

The X matrix
------------

The most critical piece of an epistasis models is the X (model) matrix.
This matrix maps genotypes to epistatic coefficients. You can read about this matrix
in this paper_.

There are two popular X matrices that exist in the epistasis literature, the
``global`` (aka background-averaged) and ``local`` (aka biochemical) model.
All epistasis models in this API takes a ``model_type`` keyword argument
that tells the model which matrix to use. Read the paper mentioned
above for more information on which model to use.

Constructing these matrices for your dataset is no easy task,
so each epistasis model can handle this construction internally. Most methods
allow you to pass one of the keywords below to construct matrices you're likely to use.

Any X matrix used by an epistasis model is also stored in the ``Xbuilt`` attribute.
This speeds up fitting algorithms that may need resample fitting many times.

These keys include:

  1. **obs** : model matrix for the observed genotypes in an attached genotype-phenotype map. These genotypes are returned by the ``genotypes`` attribute in the ``GenotypePhenotypeMap`` object.
  2. **missing** : model matrix for missing genotypes in the genotype-phenotype map. These genotypes are returned by the ``missing_genotypes`` attribute in the ``GenotypePhenotypeMap`` object.
  3. **complete** : model matrix for complete genotypes in the genotype-phenotype map.These genotypes are returned by the ``complete_genotypes`` attribute in the ``GenotypePhenotypeMap`` object.
  4. **fit** : model matrix created/used by the last ``fit`` call.
  5. **predict** : model matrix created/used by the last ``predict`` call.

.. References in this document

.. _paper: http://www.genetics.org/content/205/3/1079

List of important methods
-------------------------

Every epistasis model includes the following methods:

  * **fit** : fit the model to an attached genotype-phenotype map, *or* X and y data.
  * **predict** : predict phenotypes using the X matrix or keywords (listed above). If a keyword is used, the phenotypes are in the same order as the genotypes to the corresponding keyword.
  * **score** : the pearson coefficients between the predicted phenotypes and the given data (X/y data or attached genotype-phenotype map).
  * **thetas** : flattened array of 1) *nonlinear* parameters and 2) epistatic-coefficients estimated by model.
  * **hypothesis** : computes the phenotypes for X data given a ``thetas`` array.
  * **lnlike_of_data** : returns an array of log-likelihoods for each data point given a ``thetas`` array.
  * **lnlikelihood** : returns the total log-likelihood for X/y data given a ``thetas`` array.

List of important nonlinear attributes
--------------------------------------

The extra attributes below are attached to **nonlinear** epistasis models.

  * **Additive** : a first-order EpistasisLinearRegression used to approximate the additive effects of mutations
  * **Nonlinear** : a ``lmfit.MinizerResults`` object returned by the ``lmfit.minimize`` function for estimating the nonlinear scale in a genotype-phenotype map.
  * **Linear** : a high-order EpistasisLinearRegression used to approximate the epistatic interactions in a nonlinear genotype-phenotype map.
