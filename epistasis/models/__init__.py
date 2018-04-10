"""
A library of models to decompose high-order epistasis in genotype-phenotype
maps.
"""
from .linear import EpistasisLinearRegression, EpistasisLasso, EpistasisRidge
from .nonlinear import (EpistasisNonlinearRegression,
                        EpistasisPowerTransform)
from .classifiers import EpistasisLogisticRegression
from .pipeline import EpistasisPipeline
