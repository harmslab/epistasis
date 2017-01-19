__doc__ = """\
Simulating Genotype-Phenotype maps with high-order epistasis
============================================================

This module for simulating genotype-phenotype map data from
high-order epistatic interactions.

All simulation objects in this module inherit the `GenotypePhenotypeMap`
object from the `gpmap` package and possess an `epistasis` attribute which is
an `EpistasisMap` object.
"""
# Hide the base class
__all__ = [
    "linear",
    "multiplicative",
    "nk",
    "nonlinear"
]

# Load all Simulation classes on init.
from .nk import NkSimulation
from .linear import LinearSimulation
from .nonlinear import NonlinearSimulation
