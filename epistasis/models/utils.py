import inspect
import numpy as np
import pandas as pd
from functools import wraps
from epistasis.model_matrix_ext import get_model_matrix
from epistasis.mapping import EpistasisMap, mutations_to_sites

from gpmap.utils import genotypes_to_binary

class XMatrixException(Exception):
    """Exception Subclass for X matrix errors."""


class FittingError(Exception):
    """Exception Subclass for X matrix errors."""

def arghandler(method):
    """Handle arguments."""
    @wraps(method)
    def inner(self, **kwargs):
        # Get method name
        name = method.__name__

        # Inspect function for arguments to update.
        out = inspect.signature(method)

        # Construct kwargs from signature.
        kws = {key: val.default for key, val in out.parameters.items()}
        kws.pop('self')
        kws.pop('kwargs')

        # Update kwargs with user specified kwargs.
        kws.update(**kwargs)

        # Handle each argument
        for arg in kws:
            # Get handler function.
            handler_name = "_{}".format(arg)
            handler = getattr(self, handler_name)
            kws[arg] = handler(data=kws[arg], method=name)

        return method(self, **kws)
    return inner

def epistasis_fitter(fit_method):
    """Connect an epistasis object to the model.
    """
    @wraps(fit_method)
    def inner(self, X=None, *args, **kwargs):
        if type(X) is np.ndarray or type(X) is pd.DataFrame:
            model = fit_method(self, X=X, *args, **kwargs)

        elif X not in self.Xbuilt:
            # Map those columns to epistastalis dataframe.
            self.epistasis = EpistasisMap(
                sites=self.Xcolumns,
                order=self.order,
                model_type=self.model_type)

            # Execute fitting method
            model = fit_method(self, X=X, *args, **kwargs)

            # Link coefs to epistasis values.
            self.epistasis.values = np.reshape(self.coef_, (-1,))

        else:
            model = fit_method(self, X=X, *args, **kwargs)

        return model
    return inner
