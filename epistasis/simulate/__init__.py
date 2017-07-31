__doc__ = """\
Simulating Genotype-Phenotype maps with high-order epistasis
============================================================

This module includes objects to simulate genotype-phenotype maps with high-order
epistasis.

All simulation objects in this module inherit the `GenotypePhenotypeMap`
object from the `gpmap` package and possess an `epistasis` attribute which is
an `EpistasisMap` object.
"""
# Hide the base class
__all__ = [
    "linear",
    "nonlinear"
]

# Load all Simulation classes on init.
from .linear import LinearSimulation
from .nonlinear import NonlinearSimulation
