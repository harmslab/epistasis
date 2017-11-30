Epistasis models
================

This page provides short descriptions of each epistasis model available in this library.

Table of contents
-----------------

* EpistasisLinearRegression
* EpistasisNonLinearRegression
* EpistasisPowerTransform
* EpistasisMixedRegression


Linear, high-order epistasis model
----------------------------------

A linear, high-order epistasis model is provided through the
`EpistasisLinearRegression` class. This uses an ordinary least-squares 
regression to estimate high-order, epistatic coefficients in an arbitrary
genotype-phenotype map. Simple define the order of the model.

.. code-block:: python

  from epistasis.models import EpistasisLinearRegression
  
  wildtype = '00'
  genotypes = ['00', '01', '10', '11']
  phenotypes = ['']
  
  # Initialize the data.
  model = EpistasisLinearRegression(order=3)
  
  # Add Genotype-phenotype map data.
  model.add_data(wildtype, genotypes, phenotypes)
  
  # Fit the model.
  model.fit()


Nonlinear, high-order epistasis
-------------------------------

A nonlinear, high-order epistasis model is provided through the
`EpistasisNonlinearRegression` class. This uses nonlinear, least-squares 
regression (provided by scipy's `curve_fit`) to estimate high-order, epistatic 
coefficients in an arbitrary genotype-phenotype map. 

This models has three steps:
  1. Fit an additive, linear regression to approximate the average effect of individual mutations.
  2. Fit the nonlinear function to the observed phenotypes vs. the additive phenotypes estimated in step 1. This function is defined by the user as a callable python function
  3. Transform the phenotypes to this linear scale and fit leftover variation with high-order epistasis model.

.. code-block:: python

  import numpy as np
  from epistasis.models import EpistasisLinearRegression

  wildtype = '00'
  genotypes = ['00', '01', '10', '11']
  phenotypes = ['']

  def func(x, A):
      return np.exp(A * x)

  # Initialize the data.
  model = EpistasisLinearRegression(order=3, function=func)

  # Add Genotype-phenotype map data.
  model.add_data(wildtype, genotypes, phenotypes)

  # Fit the model.
  model.fit(A=1)


Nonlinear power transform
-------------------------

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


Handling lethal/dead phenotypes
-------------------------------

A high-order epistasis regression that first classifies genotypes as viable/nonviable (given some threshold) and then estimates epistatic coefficients in viable phenotypes.
