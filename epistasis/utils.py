__doc__ = """Submodule with handy utilities used throughout the package.
"""
# -------------------------------------------------------
# Miscellaneous Python functions for random task
# -------------------------------------------------------

import abc
import itertools as it
import numpy as np
from scipy.misc import comb
from sklearn.metrics import mean_squared_error
from collections import OrderedDict

from gpmap.utils import genotypes_to_binary
from .mapping import mutations_to_sites

from epistasis.matrix import get_model_matrix


# -------------------------------------------------------
# Custom exceptions
# -------------------------------------------------------

class SubclassException(Exception):
    """ For methods that must be implemented in a subclass. """

# -------------------------------------------------------
# Useful methods
# -------------------------------------------------------

def genotypes_to_X(wildtype, genotypes,
    order=1,
    mutations=None,
    model_type='global'):
    """Build an X matrix for a list of genotypes."""
    # Binary representation
    binary = genotypes_to_binary(wildtype, genotypes, mutations)

    # Build list of sites from genotypes.
    sites = mutations_to_sites(order, mutations)

    # X matrix
    X = get_model_matrix(binary, sites, model_type=model_type)
    return X

# -------------------------------------------------------
# Useful Classes
# -------------------------------------------------------

class DocstringMeta(abc.ABCMeta):
    """Metaclass that allows docstring 'inheritance'

    Idea taken from this thread:
    https://github.com/sphinx-doc/sphinx/issues/3140
    """
    def __new__(mcls, classname, bases, cls_dict):
        # Create a new class as expected.
        cls = abc.ABCMeta.__new__(mcls, classname, bases, cls_dict)

        # Get order of inheritance
        mro = cls.__mro__[1:]

        # Iterate through items in class.
        for name, member in cls_dict.iteritems():

            # If the item does not have a docstring, add the base class docstring.
            if not getattr(member, '__doc__'):
                for base in mro:
                    try:
                        member.__doc__ = getattr(base, name).__doc__
                        break
                    except AttributeError:
                        pass
        return cls


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
            elif self.__dict__[key] is None:
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
        unique = list(np.unique(genotypes_array[:, i]))
        if len(unique) != 1:
            mutations[i] = unique
    return mutations
