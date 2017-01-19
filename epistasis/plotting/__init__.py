__doc__ = """Plotting module for the epistasis package. The main module includes
various plotting functions, which submodules include objects for visualizing data
from epistasis model classes.
"""

from .coefs import coefs
from .correlation import corr, resid, rhist, corr_resid, corr_resid_rhist
from .fraction_explained import fraction_explained
