"""
A library of models to decompose high-order epistasis in genotype-phenotype maps.
"""
from .linear import EpistasisLinearRegression
from .power import EpistasisPowerTransform
from .nonlinear import EpistasisNonlinearRegression
from .classifiers import EpistasisLogisticRegression
from .mixed import EpistasisMixedRegression
