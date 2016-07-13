import matplotlib.pyplot as plt
import numpy as np
from seqspace.plotting import PlottingContainer, mpl_missing
from epistasis.plotting import epistasis
# ---------------------------------------------------
# Exceptions
# ---------------------------------------------------

class LogScalingException(Exception):
    """ Exception for handling log scaling problems when plotting. """

# ---------------------------------------------------
# Various plotting classes
# ---------------------------------------------------

class EpistasisPlotting(PlottingContainer):

    def __init__(self, model):
        """Plotting class for epistasis models.
        Parameters
        ----------
        model : epistasis model object
        """
        self.model = model
        super(EpistasisPlotting, self).__init__(self.model)

    def epistasis(self, figsize=(6,4), **kwargs):
        try:
            errors = [self.model.epistasis.err.lower, self.model.epistasis.err.upper]
        except AttributeError:
            errors = None
        fig, ax = epistasis(self.model.epistasis.values,
            self.model.epistasis.labels,
            errors=errors,
            logbase=self.model.logbase,
            log_transform=self.model.log_transform,
            figsize=figsize, **kwargs)
        return fig, ax
