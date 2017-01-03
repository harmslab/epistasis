__doc__ = """\
Simulating Genotype-Phenotype maps with high-order epistasis
============================================================

This module for simulating genotype-phenotype map data from
high-order epistatic interactions.

All simulation objects in this module inherit the `GenotypePhenotypeMap`
object from the `seqspace` package and possess an `epistasis` attribute which is
an `EpistasisMap` object.
"""
# Hide the base class
__all__ = [
    "additive",
    "multiplicative",
    "nk",
    "nonlinear"
]

# Load all Simulation classes on init.
from .nk import NkSimulation
from .additive import AdditiveSimulation
from .multiplicative import MultiplicativeSimulation
#from .nonlinear import NonlinearSimulation
