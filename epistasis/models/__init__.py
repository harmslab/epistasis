__doc__ = """
This module various models for calculating epistasis in genotype-phenotype maps.

Linear models
=============

`LocalEpistasisModel`
---------------------
Full linear decomposition of a genotype-phenotype map into linear epistatic
coefficients. Requires explicit choice of reference state. Epistasis is calculated
as deviation from expectation with respect to this reference state.

`GlobalEpistasisModel`
---------------------
Full linear decomposition of a genotype-phenotype map into linear epistatic
coefficients. Chooses `average` genotype as reference state to calculate epistasis.
Coefficients represent deviation from expectation with respect to this average state.

`EpistasisRegression`
--------------------
Ordinary-least-squares regression of linear epistasis model.

Nonlinear Epistasis model
=========================
"""
__all__ =  ["linear",
    "nonlinear",
    "pca",
    "regression",
    "specifier"
]

# Import principal component analysiss
from epistasis.models.pca import EpistasisPCA

# Import linear models
from epistasis.models.linear import (LinearEpistasisModel,
                                    LocalEpistasisModel,
                                    GlobalEpistasisModel)

# Import regression
from epistasis.models.regression import EpistasisRegression

# import nonlinear model
from epistasis.models.nonlinear import NonlinearEpistasisModel

# import epistasis specifier
from epistasis.models.specifier import (LinearEpistasisSpecifier,
                                        NonlinearEpistasisSpecifier)
