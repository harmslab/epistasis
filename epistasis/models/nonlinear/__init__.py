__doc__ = """\
Module for estimating epistatic coefficients in nonlinear genotype-phenotype maps.

Essentially, these models wrap linear epistasis models with a nonlinear function
that estimates how mutations 'scale' together. The simplest scaled a
assumes the effects of multiple mutations add together. However, this assumption
is not always true (or obvious). Mutations can multiply or combine in some other
nonlinear fashion.

Nonlinear epistasis models determine the scale of multiple mutations regressing
the average, independent effects of mutations in all backgrounds and comparing
the predicted phenotypes from these effects against the observed phenotypes.
The nonlinear `scale`, then, is defined as the curve that best fits this relationship.
"""
from .power import EpistasisPowerTransform
from .regression import EpistasisNonlinearRegression
