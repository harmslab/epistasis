"""
A library of models to decompose high-order epistasis in genotype-phenotype
maps.
"""
from .linear import EpistasisLinearRegression, EpistasisLasso, EpistasisRidge
from .power import EpistasisPowerTransform
from .nonlinear import (EpistasisNonlinearRegression,
                        Parameter,
                        Parameters)
from .classifiers import EpistasisLogisticRegression
from .pipeline import EpistasisPipeline
