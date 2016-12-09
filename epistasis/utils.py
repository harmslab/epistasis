__doc__ = """Submodule with handy utilities for constructing epistasis models."""
# -------------------------------------------------------
# Miscellaneous Python functions for random task
# -------------------------------------------------------

import itertools as it
import numpy as np
from scipy.misc import comb
from sklearn.metrics import mean_squared_error
from collections import OrderedDict

# -------------------------------------------------------
# Custom exceptions
# -------------------------------------------------------

class SubclassException(Exception):
    """ For methods that must be implemented in a subclass. """

# -------------------------------------------------------
# Useful methods
# -------------------------------------------------------

class Bunch:
    """Classic bunch object for constructing empty objects."""
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def update(self, **kwargs):
        """Turn a dictionary into an object with"""
        types = dict([(key, type(val)) for key, val in self.__dict__.items()])
        for key, value in kwargs.items():
            typed = types[key]
            if typed == np.ufunc:
                typed_val = value
            elif self.__dict__[key] == None:
                typed_val = value
            else:
                typed_val = types[key](value)
            setattr(self, key, typed_val)

def extract_mutations_from_genotypes(genotypes):
    """ Given a list of genotypes, infer a mutations dictionary.
    """
    genotypes_grid = [list(g) for g in genotypes]
    genotypes_array = np.array(genotypes_grid)
    (n_genotypes, n_sites) = genotypes_array.shape
    mutations = dict([(i, None) for i in range(n_sites)])
    for i in range(n_sites):
        unique = list(np.unique(genotypes_array[:,i]))
        if len(unique) != 1:
            mutations[i] = unique
    return mutations
