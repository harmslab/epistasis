import inspect
import numpy as np
import pandas as pd
from functools import wraps
from epistasis.matrix import get_model_matrix
from epistasis.mapping import EpistasisMap, mutations_to_sites

from gpmap.utils import genotypes_to_binary

class XMatrixException(Exception):
    """Exception Subclass for X matrix errors."""


class FittingError(Exception):
    """Exception Subclass for X matrix errors."""

def arghandler(method):
    """Points methods to argument handlers. Assumes each argument has a
    corresponding method attached to the object named "_{argument}". These
    methods given default values to arguments.

    Ignores self and kwargs
    """
    @wraps(method)
    def inner(self, *args, **kwargs):
        # Get method name
        name = method.__name__

        # Inspect function for arguments to update.
        out = inspect.signature(method)

        # Construct kwargs from signature.
        kws = {key: val.default for key, val in out.parameters.items()}
        kws.pop('self')

        # Try to remove kwargs.
        try:
            kws.pop('kwargs')
        except:
            pass

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
