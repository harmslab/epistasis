"""
A library of models to decompose high-order epistasis in genotype-phenotype
maps.
"""
from .linear import (EpistasisLinearRegression,
                     EpistasisLasso,
                     EpistasisRidge,
                     EpistasisElasticNet)
from .nonlinear import (EpistasisNonlinearRegression,
                        EpistasisPowerTransform,
                        EpistasisSpline)
from .classifiers import EpistasisLogisticRegression
from .pipeline import EpistasisPipeline
