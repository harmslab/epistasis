# Mapping Object for epistatic interactions int the epistasis map
#
# Author: Zach Sailer
#
# ----------------------------------------------------------
# Outside imports
# ----------------------------------------------------------

import json
import itertools as it
from functools import wraps
from collections import OrderedDict

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# Local imports
# ----------------------------------------------------------

import gpmap
from gpmap.mapping import BaseMap

def assert_epistasis(method):
    """Assert that an epistasis map has been attached to the object.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, "epistasis") is False:
            raise AttributeError(self.__name__ + " does not an epistasis attribute set yet.")
        return method(self, *args, **kwargs)
    return wrapper

def site_to_key(site, state=""):
    """Convert site to key. `state` is added to end of key."""
    if type(state) != str:
        raise Exception("`state` must be a string.")
    return ",".join([str(l) for l in site]) + state

def key_to_site(key):
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

def mutations_to_sites(order, mutations, start_order=0):
    """Build interaction sites up to nth order given a mutation alphabet.

    Parameters
    ----------
    order : int
        order of interactions
    mutations  : dict
        `mutations = { site_number : ["mutation-1", "mutation-2"] }`. If the site
        alphabet is note included, the model will assume binary
        between wildtype and derived.

    Example
    -------
    .. code-block:: python

        mutations = {
            0: ["A", "V"],
            1: ["A", "V"],
            ...
        }

    Returns
    -------
    sites : list
        list of all interaction sites for system with
        sequences of a given length and epistasis with given order.
    """
    # Convert a mutations mapping dictionary to a site mapping dictionary
    sitemap = dict()
    n_sites = 1
    for m in mutations:
        if mutations[m] is None:
            sitemap[m] = None
        else:
            sitemap[m] = list(range(n_sites, n_sites + len(mutations[m]) - 1))
            n_sites += len(mutations[m])-1

    # Include the intercept interaction?
    if start_order == 0:
        sites = [[0]]
        orders = range(1,order+1)
    else:
        sites = list()
        orders = range(start_order,order+1)

    length = len(sitemap)
    # Recursive algorithm that's difficult to follow.

    # Iterate through each order
    for o in orders:
        # Iterate through all combinations of orders with given length
        for term in it.combinations(range(length), o):
            # If any sites in `term` == None, skip this term.
            bad_term = False
            lists = []
            for i in range(len(term)):
                if sitemap[term[i]] == None:
                    bad_term = True
                    break
                else:
                    lists.append(sitemap[term[i]])
            # Else, add interactions combinations to list
            if bad_term is False:
                for r in it.product(*lists):
                    sites.append(list(r))
    return sites

class EpistasisMap(BaseMap):
    """Memory-efficient object to store/map epistatic coefficients in epistasis models.
    """
    def __init__(self, sites, order=1, model_type="global"):
        self.sites = sites
        self.order = order
        self.model_type = model_type
        self.stdeviations = None

    def to_json(self, filename):
        """Write epistasis map to json file."""
        with open(filename, "w") as f:
            data = {
                "sites" : list(self.sites),
                "values" : list(self.values),
                "order" : self.order,
                "keys" : list(self.keys),
                "model_type" : self.model_type
            }
            try:
                data.update(stdeviations=self.stdeviations)
            except AttributeError: pass
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

    @classmethod
    def from_mutations(cls, mutations, order, model_type="global"):
        """Build a mapping object for epistatic interactions."""
        # construct the mutations mapping
        sites = mutations_to_sites(order, mutations)
        self = cls(sites, order=order, model_type=model_type)
        return self

    @property
    def model_type(self):
        """Type of epistasis model used to get coefficients."""
        return self._model_type

    @property
    def df(self):
        """EpistasisMap DataFrame."""
        data = {"sites":self.keys, "values":self.values, "stdeviations":self.stdeviations}
        return pd.DataFrame(data, columns=["sites", "values", "stdeviations"])

    @property
    def n(self):
        """ Return the number of Interactions. """
        return len(self.sites)

    @property
    def order(self):
        """ Get order of epistasis in system. """
        return self._order

    @property
    def keys(self):
        """ Get the interaction keys. (type==list of str, see self._build_interaction_sites)"""
        return self._keys

    @property
    def values(self):
        """ Get the values of the interaction in the system"""
        return self._values

    @property
    def index(self):
        """ Get the interaction index in interaction matrix. """
        return self.sites.index

    @property
    def sites(self):
        """ Get the interaction sites, which describe the position of interacting mutations in
            the genotypes. (type==list of lists, see self._build_interaction_sites)
        """
        return self._sites

    @property
    def keys(self):
        """ Get interactions as string-keys. """
        if hasattr(self, '_keys'):
            return self._keys
        else:
            return np.array([site_to_key(lab) for lab in self.sites])

    @property
    def stdeviations(self):
        """Get standard deviations from model"""
        return self._stdeviations

    def get_orders(self, *orders):
        """Get epistasis of a given order."""
        return Orders(self, orders)

    # ----------------------------------------------
    # Setter Functions
    # ----------------------------------------------

    @order.setter
    def order(self, order):
        """"""
        self._order = order

    @sites.setter
    def sites(self, sites):
        """ Manually set the interactions considered in the map. Useful for building epistasis models manually. """
        self._sites = pd.Series(sites)

    @values.setter
    def values(self, values):
        """ Set the interactions of the system, set by an Epistasis model (see ..models.py)."""
        if hasattr(self, "_sites") is False:
            raise AttributeError(self.__name__ + " does not have coef sites set.")
        self._values = pd.Series(values, index=self.index)

    @keys.setter
    def keys(self, keys):
        """ Manually set keys. NEED TO do some quality control here. """
        self._keys = pd.Series(keys, index=self.index)

    @stdeviations.setter
    def stdeviations(self, stdeviations):
        """Set the standard deviations of the epistatic coefficients."""
        self._stdeviations = pd.Series(stdeviations, index=self.index)
        self.std = gpmap.errors.StandardDeviationMap(self)
        self.err = gpmap.errors.StandardErrorMap(self)

    @model_type.setter
    def model_type(self, model_type):
        types = ["global", "local"]
        if model_type not in types:
            raise Exception("Model type must be global or local")
        self._model_type = model_type

class Orders(BaseMap):
    """An object that provides API for easily calling epistasis of a given order
    in an epistasis map.
    """
    def __init__(self, epistasismap, orders):
        self._epistasismap = epistasismap
        self.orders = orders

    def __call__(self):
        """return a dictioanry"""
        return dict(zip(self.keys, self.values))

    @property
    def df(self):
        """Dataframe for orders object."""
        data = {"sites": self.sites, "values": self.values, "stdeviations": self.stdeviations}
        return pd.DataFrame(data, columns=["sites", "values","stdeviations"])

    @property
    def index(self):
        """Get indices of epistasis from this order."""
        # Check is multiple orders were given
        try:
            orders = list(iter(self.orders))
        except TypeError:
            orders = [self.orders]
        sites = self._epistasismap.sites
        x = [i for i in range(1,len(sites)) if len(sites[i]) in orders]
        # Add the zeroth element if included
        if 0 in orders:
            x = [0]+x
        return np.array(x)

    @property
    def sites(self):
        """Get epistatic sites"""
        return pd.Series([self._epistasismap.sites[int(i)] for i in self.index], index=self.index)

    @property
    def values(self):
        """Get values of epistasis for this order."""
        return pd.Series([self._epistasismap.values[int(i)] for i in self.index], index=self.index)

    @property
    def keys(self):
        """Get keys of epistasis for this order."""
        return pd.Series([self._epistasismap.keys[int(i)] for i in self.index], index=self.index)

    @property
    def stdeviations(self):
        """Get stdeviations of epistasis for this order."""
        return pd.Series(self._epistasismap.stdeviations[self.index], index=self.index)
