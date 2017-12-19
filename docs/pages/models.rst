Detailed list of models
=======================

This page provides short descriptions of each epistasis model available in this library.

EpistasisLinearRegression
-------------------------

A linear, high-order epistasis model. This uses an ordinary least-squares
regression to estimate high-order, epistatic coefficients in an arbitrary
genotype-phenotype map. Simple define the order of the model.

.. code-block:: python

  from gpmap import GenotypePhenotypeMap
  from epistasis.models import EpistasisLinearRegression

  wildtype = 'AA'
  genotypes = ['AA', 'AT', 'TA', 'TT']
  phenotypes = [0.1, 0.2, 0.7, 1.2]

  # Read genotype-phenotype map.
  gpm = GenotypePhenotypeMap(wildtype, genotypes, phenotypes)

  # Initialize the data.
  model = EpistasisLinearRegression(order=2)

  # Add Genotype-phenotype map data.
  model.add_gpm(gpm)

  # Fit the model.
  model.fit()

EpistasisLasso
--------------

A L1-norm epistasis model for estimating sparse epistatic coefficients. The
optimization function imposes a penalty on the number of coefficients and finds
the model that maximally explains the data while using the fewest coefficients.

.. code-block:: python

  from gpmap import GenotypePhenotypeMap
  from epistasis.models import EpistasisLinearRegression

  wildtype = 'AA'
  genotypes = ['AA', 'AT', 'TA', 'TT']
  phenotypes = [0.1, 0.2, 0.7, 1.2]

  # Read genotype-phenotype map.
  gpm = GenotypePhenotypeMap(wildtype, genotypes, phenotypes)

  # Initialize the data.
  model = EpistasisLasso(order=2)

  # Add Genotype-phenotype map data.
  model.add_gpm(gpm)

  # Fit the model.
  model.fit()


EpistasisNonlinearRegression
----------------------------

A nonlinear, high-order epistasis model. This uses nonlinear, least-squares
regression (provided by ``lmfit``) to estimate high-order, epistatic
coefficients in an arbitrary genotype-phenotype map.

This models has three steps:
  1. Fit an additive, linear regression to approximate the average effect of individual mutations.
  2. Fit the nonlinear function to the observed phenotypes vs. the additive phenotypes estimated in step 1. This function is defined by the user as a callable python function
  3. Transform the phenotypes to this linear scale and fit leftover variation with high-order epistasis model.

.. code-block:: python

  from gpmap import GenotypePhenotypeMap
  from epistasis.models import EpistasisLinearRegression

  wildtype = 'AA'
  genotypes = ['AA', 'AT', 'TA', 'TT']
  phenotypes = [0.1, 0.2, 0.7, 1.2]

  # Read genotype-phenotype map.
  gpm = GenotypePhenotypeMap(wildtype, genotypes, phenotypes)

  def func(x, A):
      return np.exp(A * x)

  def reverse(y, A):
      return np.log(x) / A

  # Initialize the data.
  model = EpistasisNonlinearRegression(order=2, function=func, reverse=reverse)

  # Add Genotype-phenotype map data.
  model.add_gpm(gpm)

  # Fit the model.
  model.fit(A=1)


EpistasisNonlinearLasso
-----------------------

A nonlinear, high-order epistasis model. This uses nonlinear, least-squares
regression (provided by ``lmfit``) to estimate high-order, epistatic
coefficients in an arbitrary genotype-phenotype map.

This models has three steps:
  1. Fit an additive, linear regression to approximate the average effect of individual mutations.
  2. Fit the nonlinear function to the observed phenotypes vs. the additive phenotypes estimated in step 1. This function is defined by the user as a callable python function
  3. Transform the phenotypes to this linear scale and fit leftover variation with an EpistasisLasso.

.. code-block:: python

  from gpmap import GenotypePhenotypeMap
  from epistasis.models import EpistasisLinearRegression

  wildtype = 'AA'
  genotypes = ['AA', 'AT', 'TA', 'TT']
  phenotypes = [0.1, 0.2, 0.7, 1.2]

  # Read genotype-phenotype map.
  gpm = GenotypePhenotypeMap(wildtype, genotypes, phenotypes)

    def func(x, A):
        return np.exp(A * x)

    def reverse(y, A):
        return np.log(x) / A

    # Initialize the data.
    model = EpistasisNonlinearLasso(order=3, function=func, reverse=reverse)

    # Add Genotype-phenotype map data.
    model.add_gpm(gpm)

    # Fit the model.
    model.fit(A=1)


EpistasisPowerTransform
-----------------------

Use power-transform function, via nonlinear least-squares regression, to
estimate epistatic coefficients and the nonlinear scale in a nonlinear
genotype-phenotype map.

Like the nonlinear model, this model has three steps:
  1. Fit an additive, linear regression to approximate the average effect of individual mutations.
  2. Fit the nonlinear function to the observed phenotypes vs. the additive phenotypes estimated in step 1.
  3. Transform the phenotypes to this linear scale and fit leftover variation with high-order epistasis model.

Methods are described in the following publication:

    Sailer, Z. R. & Harms, M. J. 'Detecting High-Order Epistasis in Nonlinear
    Genotype-Phenotype Maps'. Genetics 205, 1079-1088 (2017).

.. code-block:: python

    from gpmap import GenotypePhenotypeMap
    from epistasis.models import EpistasisLinearRegression

    wildtype = 'AA'
    genotypes = ['AA', 'AT', 'TA', 'TT']
    phenotypes = [0.1, 0.2, 0.7, 1.2]

    # Read genotype-phenotype map.
    gpm = GenotypePhenotypeMap(wildtype, genotypes, phenotypes)

    # Initialize the data.
    model = EpistasisPowerTransform(order=3)

    # Add Genotype-phenotype map data.
    model.add_gpm(gpm)

    # Fit the model.
    model.fit(lmbda=1, A=1, B=1)


EpistasisPowerLasso
-------------------

Use power-transform function, via nonlinear least-squares regression, to
estimate epistatic coefficients and the nonlinear scale in a nonlinear
genotype-phenotype map.

Like the nonlinear model, this model has three steps:
  1. Fit an additive, linear regression to approximate the average effect of individual mutations.
  2. Fit the nonlinear function to the observed phenotypes vs. the additive phenotypes estimated in step 1.
  3. Transform the phenotypes to this linear scale and fit leftover variation with an EpistasisLasso.


.. code-block:: python

    from gpmap import GenotypePhenotypeMap
    from epistasis.models import EpistasisLinearRegression

    wildtype = 'AA'
    genotypes = ['AA', 'AT', 'TA', 'TT']
    phenotypes = [0.1, 0.2, 0.7, 1.2]

    # Read genotype-phenotype map.
    gpm = GenotypePhenotypeMap(wildtype, genotypes, phenotypes)

    # Initialize the data.
    model = EpistasisPowerTransformLasso(order=3)

    # Add Genotype-phenotype map data.
    model.add_gpm(gpm)

    # Fit the model.
    model.fit(lmbda=1, A=1, B=1)


EpistasisLogisticRegression
---------------------------

A high-order epistasis regression that classifies genotypes as viable/nonviable (given some threshold).

.. code-block:: python

  from epistasis.models import EpistasisLogisticRegression

  wildtype = '00'
  genotypes = ['00', '01', '10', '11']
  phenotypes = [0, .2, .1, 1]

  # Initialize the data.
  model = EpistasisLogisticRegression(order=1, threshold=.1)

  # Add Genotype-phenotype map data.
  model.add_data(wildtype, genotypes, phenotypes)

  # Fit the model.
  model.fit()


EpistasisMixedRegression
---------------------------

A high-order epistasis regression that first classifies genotypes as viable/nonviable (given some threshold), then
fits an epistasis model to estimate epistatic coefficients.


.. code-block:: python


  from gpmap import GenotypePhenotypeMap

  from epistasis.models import (EpistasisMixedRegression,
                              EpistasisPowerTransform,
                              EpistasisLogisticRegression)

  wildtype = 'AA'
  genotypes = ['AA', 'AT', 'TA', 'TT']
  phenotypes = [0.1, 0.2, 0.7, 1.2]

  # Read genotype-phenotype map.
  gpm = GenotypePhenotypeMap(wildtype, genotypes, phenotypes)

  # Construct a classifier and an epistasis model
  classifier = EpistasisLogisticRegression(order=1, threshold=.2, model_type='global')
  model = EpistasisPowerTransform(order=2, model_type='global', alpha=.1, lmbda=1, A=100,B=-1)

  # Initialize a Mixed regression that links the classifier and epistasis model.
  model = EpistasisMixedRegression(classifier, model)

  # Add the genotype-phenotype map to the mixed model
  model.add_gpm(gpm)

  # Fit the model.
  model.fit()
