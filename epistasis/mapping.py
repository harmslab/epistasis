# Mapping Object for epistatic interactions int the epistasis map
#
# Author: Zach Sailer
#
# ----------------------------------------------------------
# Outside imports
# ----------------------------------------------------------

from functools import wraps
import itertools as it
import numpy as np
import json
from collections import OrderedDict

# ----------------------------------------------------------
# Local imports
# ----------------------------------------------------------

import gpmap
from gpmap.base import BaseMap

def assert_epistasis(method):
    """Assert that an epistasis map has been attached to the object.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, "epistasis") is False:
            raise AttributeError(self.__name__ + " does not an epistasis attribute set yet.")
        return method(self, *args, **kwargs)
    return wrapper

def label_to_key(label, state=""):
    """ Convert interaction label to key. `state` is added to end of key."""
    if type(state) != str:
        raise Exception("`state` must be a string.")
    return ",".join([str(l) for l in label]) + state

def key_to_label(key):
    """ Convert an interaction key to label."""
    return [int(k) for k in key.split(",")]

def genotype_coeffs(genotype, order=None):
    """List the possible epistatic coefficients (as label form) for a binary genotype
    up to a given order.
    """
    if order is None:
        order = len(genotype)
    length = len(genotype)
    mutations = [i + 1 for i in range(length) if genotype[i] == "1"]
    params = [[0]]
    for o in range(1, order+1):
        params += [list(z) for z in it.combinations(mutations, o)]
    return params

def mutations_to_coeffs(mutations):
    """Write a dictionary that maps mutations dictionary to indices in dummy
    variable matrix.

    Parameters
    ----------
    mutations : dict
        mapping each site to their accessible mutations alphabet.
        mutations = {site_number : alphabet} If site does not mutate,
        value should be None.

    Returns
    -------
    mutations : dict
        `mutations = { site_number : indices }`. If the site alphabet is
        note included, the model will assume binary between wildtype and derived.

    Example
    -------
    .. code-block:: python

        mutations = {
            0: [indices],
            1: [indices],
            ...
        }
    """
    param_map = dict()
    n_sites = 1
    for m in mutations:
        if mutations[m] is None:
            param_map[m] = None
        else:
            param_map[m] = list(range(n_sites, n_sites + len(mutations[m]) - 1))
            n_sites += len(mutations[m])-1
    return param_map

def build_model_coeffs(order, mutations, start_order=0):
    """ Build interaction labels up to nth order given a mutation alphabet.

    Parameters
    ----------
    order : int
        order of interactions
    mutations  : dict
        `mutations = { site_number : indices }`. If the site
        alphabet is note included, the model will assume binary
        between wildtype and derived.

    Example
    -------
    .. code-block:: python

        mutations = {
            0: [indices],
            1: [indices],
            ...
        }

    Returns
    -------
    interactions : list
        list of all interaction labels for system with
        sequences of a given length and epistasis with given order.
    """
    # Include the intercept interaction?
    if start_order == 0:
        interactions = [[0]]
        orders = range(1,order+1)
    else:
        interactions = list()
        orders = range(start_order,order+1)

    length = len(mutations)
    # Recursive algorithm that's difficult to follow.

    # Iterate through each order
    for o in orders:
        # Iterate through all combinations of orders with given length
        for term in it.combinations(range(length), o):
            # If any sites in `term` == None, skip this term.
            bad_term = False
            lists = []
            for i in range(len(term)):
                if mutations[term[i]] == None:
                    bad_term = True
                    break
                else:
                    lists.append(mutations[term[i]])
            # Else, add interactions combinations to list
            if bad_term is False:
                for r in it.product(*lists):
                    interactions.append(list(r))
    return interactions

class EpistasisMap(BaseMap):
    """Efficient mapping object for epistatic coefficients in an EpistasisModel.
    """
    def to_json(self, filename):
        """Write epistasis map to json file."""
        with open(filename, "w") as f:
            data = {
                "labels" : list(self.labels),
                "values" : list(self.values),
                "order" : self.order,
                "keys" : list(self.keys),
                "model_type" : self.model_type
            }
            try:
                data.update(stdeviations=self.stdeviations)
            except AttributeError:
                pass
            json.dump(data, f)

    def from_json(self, filename):
        """Read epistatic coefs from json file.
        """
        with open(filename, "r") as f:
            data = json.load(f)
        # Values must be set last
        vals = data.pop("values")
        for key, val in data.items():
            if type(val) == list:
                val = np.array(val)
            setattr(self, key, val)
        # Now set values.
        setattr(self, "values", vals)
        self._getorder = dict([(i, Order(self, i)) for i in range(0, self.order+1)])

    def _from_labels(self, labels, model_type="global"):
        """Set coef labels of an epistasis map instance.
        """
        self.labels = labels
        self.order = max([len(l) for l in labels])
        self._getorder = dict([(i, Order(self, i)) for i in range(0, self.order+1)])
        self.model_type = model_type

    @classmethod
    def from_labels(cls, labels, model_type="global"):
        """Construct an EpistasisMap instance from labels.
        """
        self = cls()
        self._from_labels(labels, model_type=model_type)
        return self

    def _from_mutations(self, mutations, order, model_type="global"):
        """Set coef labels from mutations and order.
        """
        self.order = order
        self._labels = build_model_coeffs(
            self.order,
            mutations_to_coeffs(mutations)
        )
        self._getorder = dict([(i, Order(self, i)) for i in range(0, self.order+1)])
        self.model_type = model_type

    @classmethod
    def from_mutations(cls, mutations, order, model_type="global"):
        """Build a mapping object for epistatic interactions."""
        # construct the mutations mapping
        self = cls()
        self._from_mutations(mutations, order, model_type=model_type)
        return self

    @property
    def model_type(self):
        """Type of epistasis model used to get coefficients."""
        return self._model_type

    @property
    def n(self):
        """ Return the number of Interactions. """
        return len(self.labels)

    @property
    def order(self):
        """ Get order of epistasis in system. """
        return self._order

    @property
    def keys(self):
        """ Get the interaction keys. (type==list of str, see self._build_interaction_labels)"""
        return self._keys

    @property
    def values(self):
        """ Get the values of the interaction in the system"""
        return self._values

    @property
    def indices(self):
        """ Get the interaction index in interaction matrix. """
        return self._indices

    @property
    def labels(self):
        """ Get the interaction labels, which describe the position of interacting mutations in
            the genotypes. (type==list of lists, see self._build_interaction_labels)
        """
        return self._labels

    @property
    def keys(self):
        """ Get interactions as string-keys. """
        if hasattr(self, '_keys'):
            return self._keys
        else:
            return np.array([label_to_key(lab) for lab in self.labels])

    @property
    def genotypes(self):
        """ Get the interaction genotype. """
        elements = ['w.t.']
        for label in self._labels[1:]:
            elements.append(self._label_to_genotype(label))
        return elements

    @property
    def stdeviations(self):
        """Get standard deviations from model"""
        return self._stdeviations

    @property
    def getorder(self):
        """Get epistasis of a given order."""
        return self._getorder

    # ----------------------------------------------
    # Setter Functions
    # ----------------------------------------------

    @order.setter
    def order(self, order):
        """"""
        self._order = order

    @labels.setter
    def labels(self, labels):
        """ Manually set the interactions considered in the map. Useful for building epistasis models manually. """
        self._labels = labels
        self._indices = np.arange(0, len(self.labels))

    @values.setter
    def values(self, values):
        """ Set the interactions of the system, set by an Epistasis model (see ..models.py)."""
        if hasattr(self, "_labels") is False:
            raise AttributeError(self.__name__ + " does not have coef labels set.")
        elif len(self._labels) != len(values):
            raise Exception("Length of `values` must much length of `labels`.")
        self._values = values

    @keys.setter
    def keys(self, keys):
        """ Manually set keys. NEED TO do some quality control here. """
        self._keys = keys

    @stdeviations.setter
    def stdeviations(self, stdeviations):
        """Set the standard deviations of the epistatic coefficients."""
        self._stdeviations = stdeviations
        self.std = gpmap.errors.StandardDeviationMap(self)
        self.err = gpmap.errors.StandardErrorMap(self)

    @model_type.setter
    def model_type(self, model_type):
        types = ["global", "local"]
        if model_type not in types:
            raise Exception("Model type must be global or local")
        self._model_type = model_type

class Order(BaseMap):
    """An object that provides API for easily calling epistasis of a given order
    in an epistasis map.
    """
    def __init__(self, epistasismap, order):
        self._epistasismap = epistasismap
        self.order = order

    @property
    def indices(self):
        """Get indices of epistasis from this order."""
        labels = self._epistasismap.labels
        if self.order == 0:
            return np.array([0])
        x = [i for i in range(len(labels)) if len(labels[i]) == self.order]
        # Remove the zeroth element
        if self.order == 1:
            return np.array(x[1:])
        else:
            return np.array(x)

    @property
    def labels(self):
        """Get epistatic labels"""
        return [self._epistasismap.labels[int(i)] for i in self.indices]

    @property
    def values(self):
        """Get values of epistasis for this order."""
        return [self._epistasismap.values[int(i)] for i in self.indices]

    @property
    def keys(self):
        """Get keys of epistasis for this order."""
        return [self._epistasismap.keys[int(i)] for i in self.indices]

    @property
    def stdeviations(self):
        """Get stdeviations of epistasis for this order."""
        return self._epistasismap.stdeviations[self.indices]
