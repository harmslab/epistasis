"""
A library of models to decompose high-order epistasis in genotype-phenotype
maps.
"""
# Import linear models
from .linear import (EpistasisLinearRegression,
                     EpistasisLasso,
                     EpistasisRidge,
                     EpistasisElasticNet)

# Import nonlinear models
from .nonlinear import (EpistasisNonlinearRegression,
                        EpistasisPowerTransform,
                        EpistasisSpline)

# Import classifiers
from .classifiers import (EpistasisLogisticRegression,
                          EpistasisGaussianMixture,
                          EpistasisGaussianProcess)

# Import Pipeline object fro stitching models.
from .pipeline import EpistasisPipeline
