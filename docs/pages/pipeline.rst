Building a pipeline
===================

The ``EpistasisPipeline`` object allows you to link epistasis models in series.
Define each mode and add them to a pipeline. When ``fit`` is called,
this object runs a cascade of fit_transforms.

This is particularly useful if you need to remove nonlinearity in a genotype-phenotype
map before fitting high-order epistasis (see this paper_).

.. _paper: http://www.genetics.org/content/205/3/1079

``EpistasisPipeline`` inherits Python's ``list`` type. This means you can
append, prepend, pop, etc. from the pipeline after initialization. Each model is
fit in the order it appears in the pipeline.

Simple Example
--------------

In the example below, the power transform **linearizes** the map, then fits specific **high-order epistasis** on the linear scale. The fitted model
is then used to predict the phenotype of an unknown genotype.

.. code-block:: python

  from epistasis import EpistasisPipeline
  from epistasis.models import (EpistasisPowerTransform,
                                EpistasisLinearRegression)

  # Define genotype-phenotype map.
  gpm = GenotypePhenotyeMap(
    wildtype='AA'
    genotypes=['AA', 'AV','VV'],  # Note that we're missing the 'VA' genotype
    phenotypes=[0, .5, 1]
  )

  # Construct pipeline.
  model = EpistasisPipeline([
    EpistasisPowerTransform(lmbda=1, A=0, B=0),
    EpistasisLinearRegression(order=2)
  ])

  # Fit pipeline.
  model.fit()

  # Predict missing phenotype of missing genotype.
  model.predict(['VA'])
