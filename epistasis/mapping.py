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


def assert_epistasis(method):
    """Assert that an epistasis map has been attached to the object.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, "epistasis") is False:
            raise AttributeError(
                self.__name__ + " does not an epistasis attribute set yet.")
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
    """List the possible epistatic coefficients (as label form) for a binary
    genotype up to a given order.
    """
    if order is None:
        order = len(genotype)
    length = len(genotype)
    mutations = [i + 1 for i in range(length) if genotype[i] == "1"]
    params = [[0]]
    for o in range(1, order + 1):
        params += [list(z) for z in it.combinations(mutations, o)]
    return params


def mutations_to_sites(order, mutations, start_order=0):
    """Build interaction sites up to nth order given a mutation alphabet.

    Parameters
    ----------
    order : int
        order of interactions
    mutations  : dict
        `mutations = { site_number : ["mutation-1", "mutation-2"] }`.
        If the site alphabet is note included, the model will assume binary
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
            n_sites += len(mutations[m]) - 1

    # Include the intercept interaction?
    if start_order == 0:
        sites = [[0]]
        orders = range(1, order + 1)
    else:
        sites = list()
        orders = range(start_order, order + 1)

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
                if sitemap[term[i]] is None:
                    bad_term = True
                    break
                else:
                    lists.append(sitemap[term[i]])
            # Else, add interactions combinations to list
            if bad_term is False:
                for r in it.product(*lists):
                    sites.append(list(r))

    return sites


class EpistasisMap(object):
    """DataFrame for epistatic interactions.
    """
    def __init__(self, sites, order=1, values=None, model_type="global"):
        self.order = order
        self.model_type = model_type
        data = {
            'sites': sites,
            'values': values,
        }
        self.data = pd.DataFrame(data)

    def map(self, attr1, attr2):
        """Dictionary that maps attr1 to attr2."""
        return dict(zip(getattr(self, attr1), getattr(self, attr2)))

    @classmethod
    def read_dataframe(cls, df, model_type="global"):
        """Create an epistasis model from dataframe"""
        sites = df['sites']
        values = df['values']
        order = max([len(site) for site in sites])
        self = cls(sites, order=order, values=values, model_type=model_type)
        return self

    def to_dict(self):
        """Get data as dictionary."""
        return self.data.to_dict('list')

    def to_csv(self, filename):
        """Write data to a csv file."""
        self.data.to_csv(filename)

    def to_excel(self, filename):
        """Write data to excel file."""
        self.data.to_excel(filename)

    @property
    def model_type(self):
        """Type of epistasis model used to get coefficients."""
        return self._model_type

    @property
    def n(self):
        """ Return the number of Interactions. """
        return len(self.data.sites)

    @property
    def order(self):
        """ Get order of epistasis in system. """
        return self._order

    @property
    def values(self):
        """ Get the values of the interaction in the system"""
        return self.data['values'].values

    @property
    def index(self):
        """ Get the interaction index in interaction matrix. """
        return self.data.index

    @property
    def sites(self):
        """ Get the interaction sites, which describe the position of
        interacting mutations in the genotypes. (type==list of lists,
        see self._build_interaction_sites)
        """
        return self.data.sites.values

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

    @values.setter
    def values(self, values):
        """ Manually set keys. NEED TO do some quality control here. """
        self.data.values = values

    @model_type.setter
    def model_type(self, model_type):
        types = ["global", "local"]
        if model_type not in types:
            raise Exception("Model type must be global or local")
        self._model_type = model_type


class Orders(map):
    """An object that provides API for easily calling epistasis of a given order
    in an epistasis map.
    """

    def __init__(self, epistasismap, orders):
        self._epistasismap = epistasismap
        self.orders = orders

    def __call__(self):
        """return a dictionary"""
        return dict(zip(self.keys, self.values))

    @property
    def df(self):
        """Dataframe for orders object."""
        data = {"sites": self.sites, "values": self.values,
                "stdeviations": self.stdeviations}
        return pd.DataFrame(data, columns=["sites", "values", "stdeviations"])

    @property
    def index(self):
        """Get indices of epistasis from this order."""
        # Check is multiple orders were given
        try:
            orders = list(iter(self.orders))
        except TypeError:
            orders = [self.orders]
        sites = self._epistasismap.sites
        x = [i for i in range(1, len(sites)) if len(sites[i]) in orders]
        # Add the zeroth element if included
        if 0 in orders:
            x = [0] + x
        return np.array(x)

    @property
    def sites(self):
        """Get epistatic sites"""
        return pd.Series([self._epistasismap.sites[int(i)]
                          for i in self.index], index=self.index)

    @property
    def values(self):
        """Get values of epistasis for this order."""
        return pd.Series([self._epistasismap.values[int(i)]
                          for i in self.index], index=self.index)
