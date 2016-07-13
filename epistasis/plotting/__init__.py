__doc__ = """Plotting module for the epistasis package. The main module includes
various plotting functions, which submodules include objects for visualizing data
from epistasis model classes.
"""
from .epistasis import epistasis
from .stats import (correlation, residuals, magnitude_vs_order)
from .pca import principal_components
