Detailed list of models
=======================

This page lists all models included in the Epistasis Package.

* EpistasisLinearRegression_: estimate epistatic coefficents in a linear genotype-phenotype map.
* EpistasisRidge_: estimate epistatic coefficients using L2-regularization in a linear genotype-phenotype map
* EpistasisLasso_: estimate *sparse* epistatic coefficients using L1-regularization in a linear genotype-phenotype map
* EpistasisElasticNet_: estimate *sparse* epistatic coefficients, mixing L1- and L2-regularization, in a linear genotype-phenotype map
* EpistasisNonlinearRegression_: estimates nonlinear scale in genotype-phenotype map using an arbitrary defined nonlinear function.
* EpistasisSpline_: estimates nonlinear scale in genotype-phenotype map using a spline.
* EpistasisPowerTransform_: estimates nonlinear scale in genotype-phenotype map using a power transform.
* EpistasisLogisticRegression_: use logistic regression to classify phenotypes as dead/alive.
* EpistasisEnsembleRegression_: use a statistical ensemble of "states" to decompose variation in a genotype-phenotype map.

.. _EpistasisLinearRegression: models.html#epistasislinearregression
.. _EpistasisRidge: models.html#epistasisridge
.. _EpistasisLasso: models.html#epistasislasso
.. _EpistasisElasticNet: models.html#epistasisnet
.. _EpistasisNonlinearRegression: models.html#epistasisnonlinearregression
.. _EpistasisSpline: models.html#epistasisspline
.. _EpistasisPowerTransform: models.html#epistasispowertransform
.. _EpistasisLogisticRegression: models.html#epistasislogisticregression
.. _EpistasisMixedRegression: models.html#epistasismixedregression
.. _EpistasisEnsembleRegression: models.html#epistasisensembleregression


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

EpistasisRidge
--------------

A L2-norm epistasis model for estimating sparse epistatic coefficients. The
optimization function imposes a penalty on the number of coefficients and finds
the model that maximally explains the data while using the fewest coefficients.

.. code-block:: python

  from gpmap import GenotypePhenotypeMap
  from epistasis.models import EpistasisRidge

  wildtype = 'AA'
  genotypes = ['AA', 'AT', 'TA', 'TT']
  phenotypes = [0.1, 0.2, 0.7, 1.2]

  # Read genotype-phenotype map.
  gpm = GenotypePhenotypeMap(wildtype, genotypes, phenotypes)

  # Initialize the data.
  model = EpistasisRidge(order=2, alpha=0.1)

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
  from epistasis.models import EpistasisLasso

  wildtype = 'AA'
  genotypes = ['AA', 'AT', 'TA', 'TT']
  phenotypes = [0.1, 0.2, 0.7, 1.2]

  # Read genotype-phenotype map.
  gpm = GenotypePhenotypeMap(wildtype, genotypes, phenotypes)

  # Initialize the data.
  model = EpistasisLasso(order=2, alpha=0.1)

  # Add Genotype-phenotype map data.
  model.add_gpm(gpm)

  # Fit the model.
  model.fit()

EpistasisElasticNet
-------------------

A L1-norm+L2-norm epistasis model for estimating sparse epistatic coefficients. The
optimization function imposes a penalty on the number of coefficients and finds
the model that maximally explains the data while using the fewest coefficients.

.. code-block:: python

  from gpmap import GenotypePhenotypeMap
  from epistasis.models import EpistasisElasticNet

  wildtype = 'AA'
  genotypes = ['AA', 'AT', 'TA', 'TT']
  phenotypes = [0.1, 0.2, 0.7, 1.2]

  # Read genotype-phenotype map.
  gpm = GenotypePhenotypeMap(wildtype, genotypes, phenotypes)

  # Initialize the data.
  model = EpistasisElasticNet(order=2, alpha=0.1)

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
  model = EpistasisNonlinearRegression(function=func, A=1)

  # Add Genotype-phenotype map data.
  model.add_gpm(gpm)

  # Fit the model.
  model.fit()


EpistasisSpline
---------------

Use Spline function, via nonlinear least-squares regression, to
estimate epistatic coefficients and the nonlinear scale in a nonlinear
genotype-phenotype map.

Like the nonlinear model, this model has three steps:
  1. Fit an additive, linear regression to approximate the average effect of individual mutations.
  2. Fit the nonlinear function to the observed phenotypes vs. the additive phenotypes estimated in step 1.

.. code-block:: python

    from gpmap import GenotypePhenotypeMap
    from epistasis.models import EpistasisSpline

    wildtype = 'AA'
    genotypes = ['AA', 'AT', 'TA', 'TT']
    phenotypes = [0.1, 0.2, 0.7, 1.2]

    # Read genotype-phenotype map.
    gpm = GenotypePhenotypeMap(wildtype, genotypes, phenotypes)

    # Initialize the data.
    model = EpistasisSpline(k=3)

    # Add Genotype-phenotype map data.
    model.add_gpm(gpm)

    # Fit the model.
    model.fit()


EpistasisPowerTransform
-----------------------

Use power-transform function, via nonlinear least-squares regression, to
estimate epistatic coefficients and the nonlinear scale in a nonlinear
genotype-phenotype map.

Like the nonlinear model, this model has three steps:
  1. Fit an additive, linear regression to approximate the average effect of individual mutations.
  2. Fit the nonlinear function to the observed phenotypes vs. the additive phenotypes estimated in step 1.

Methods are described in the following publication:

    Sailer, Z. R. & Harms, M. J. 'Detecting High-Order Epistasis in Nonlinear
    Genotype-Phenotype Maps'. Genetics 205, 1079-1088 (2017).

.. code-block:: python

    from gpmap import GenotypePhenotypeMap
    from epistasis.models import EpistasisPowerTransform

    wildtype = 'AA'
    genotypes = ['AA', 'AT', 'TA', 'TT']
    phenotypes = [0.1, 0.2, 0.7, 1.2]

    # Read genotype-phenotype map.
    gpm = GenotypePhenotypeMap(wildtype, genotypes, phenotypes)

    # Initialize the data.
    model = EpistasisPowerTransform(lmbda=1, A=1, B=1)

    # Add Genotype-phenotype map data.
    model.add_gpm(gpm)

    # Fit the model.
    model.fit()


EpistasisLogisticRegression
---------------------------

A high-order epistasis regression that classifies genotypes as viable/nonviable (given some threshold).

.. code-block:: python

  from epistasis.models import EpistasisLogisticRegression

  wildtype = 'AA'
  genotypes = ['AA', 'AT', 'TA', 'TT']
  phenotypes = [0, .2, .1, 1]

  # Read genotype-phenotype map.
  gpm = GenotypePhenotypeMap(wildtype, genotypes, phenotypes)

  # Initialize the data.
  model = EpistasisLogisticRegression(threshold=.1)

  # Add Genotype-phenotype map data.
  model.add_gpm(gpm)

  # Fit the model.
  model.fit()


EpistasisEnsembleRegression
---------------------------
A regression object that models phenotypes as a statistical (Boltmann-weighted)
average of "states". Mutations are modeled as having different effects in each
state.

.. math::

    P = \text{ln} ( \sum_{x=\{\text{A,B,...}\}} - \text{exp}(\beta_{0; x} + \beta_{1; x} + ... + \beta_{1,2; x}+ ...) )

.. code-block:: python


    from gpmap import GenotypePhenotypeMap
    from epistasis.models import EpistasisEnsembleRegression

    wildtype = 'AA'
    genotypes = ['AA', 'AT', 'TA', 'TT']
    phenotypes = [0.1, 0.2, 0.7, 1.2]

    # Read genotype-phenotype map.
    gpm = GenotypePhenotypeMap(wildtype, genotypes, phenotypes)

    # Initialize the data.
    model = EpistasisEnsembleRegression(order=1, nstates=1)

    # Add Genotype-phenotype map data.
    model.add_gpm(gpm)

    # Fit the model.
    model.fit()

    # Print effects in state A.
    print(model.state_A.epistasis.values)
