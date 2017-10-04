__doc__ = """\
Module containing plots that are common when analyzing epistasis.
"""

from .coefs import coefs
from .correlation import corr, resid, rhist, corr_resid, corr_resid_rhist
from .fraction_explained import fraction_explained

from . import nonlinear
from . import mixed
