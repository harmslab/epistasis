__doc__ = """
Module for constructing simulated genotype-phenotype maps from high-order epistasis.
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
from .nonlinear import NonlinearSimulation
