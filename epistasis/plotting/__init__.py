__doc__ = """Plotting module for the epistasis package. The main module includes
various plotting functions, which submodules include objects for visualizing data
from epistasis model classes.
"""
import matplotlib.pyplot as _plt

try:
    import seaborn as _sns
    _sns.reset_orig()

except ImportError:
    import warnings as _warnings
    raise _warnings.WarningMessage("This module requires seaborn. Please install "
    "seaborn before using the plotting features in this module.")

from . import plotting.coefs as _coefs
from functools import wraps as _wraps
from . import plotting.fraction_explained import fraction_explained

@_wraps(_coefs)
def coefs(labels, *args, **kwargs):
    N = max([len(l) for l in labels])
    order_colors = _sns.color_palette("husl", n_colors=N)
    coefs(labels, order_colors=order_colors, *args, **kwargs)
