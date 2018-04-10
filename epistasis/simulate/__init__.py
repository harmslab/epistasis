"""

A module for simulating genotype-phenotype maps with high-order epistasis.

"""
# Hide the base class
__all__ = [
    "linear",
    "nonlinear"
]

# Load all Simulation classes on init.
from .linear import LinearSimulation
from .power import PowerScaleSimulation
