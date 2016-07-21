# Mapping Object for epistatic interactions int the epistasis map
#
# Author: Zach Sailer
#
# ----------------------------------------------------------
# Outside imports
# ----------------------------------------------------------

import itertools as it
import numpy as np
from collections import OrderedDict

# ----------------------------------------------------------
# Local imports
# ----------------------------------------------------------

from seqspace.base import BaseMap
from seqspace.errors import StandardErrorMap, StandardDeviationMap
from epistasis.utils import (params_index_map,
    build_model_params,
    label_to_key)

class TransformEpistasisMap(BaseMap):
    """Mapping object that log transforms an EpistasisMap.

    Parameters
    ----------
    EpistasisMap : EpistasisMap object
        Map to log transform
    """
    def __init__(self, EpistasisMap):
        self._epistasis = EpistasisMap
        self.transformed = True
        self.std = StandardDeviationMap(self)
        self.err = StandardErrorMap(self)

    @property
    def order(self):
        return self._epistasis.order

    @property
    def keys(self):
        return self._epistasis.keys

    @property
    def labels(self):
        return self._epistasis.labels

    @property
    def logbase(self):
        """Get base of logarithm for tranformed epistasis"""
        return self._epistasis.logbase

    @property
    def values(self):
        """ Get the values of the interaction in the system"""
        return self.logbase(self._epistasis.values)

    @property
    def stdeviations(self):
        """Get the standard deviations of the epistasis coefficients."""
        return self._stdeviations

    @property
    def n_replicates(self):
        """Get number of replicates for each observable."""
        return self._epistasis.n_replicates

    @property
    def getorder(self):
        return dict([(i, Order(self, i)) for i in range(1,self.order+1)])


class EpistasisMap(BaseMap):

    def __init__(self, GenotypePhenotypeMap):
        """ Mapping object for indexing and tracking interactions in an
        epistasis map object.

        Parameters
        ----------
        GenotypePhenotypeMap : seqspace.gpm.GenotypePhenotypeMap
            Epistasis Model to attach
        """
        self._gpm = GenotypePhenotypeMap
        self.transformed = False
        if self._gpm.log_transform:
            self.log = TransformEpistasisMap(self)
        self.std = StandardDeviationMap(self)
        self.err = StandardErrorMap(self)

    def build(self):
        """Build a mapping object for epistatic interactions."""
        # construct the mutations mapping
        self._params = params_index_map(self._gpm.mutations)
        self._labels = build_model_params(
            self.length,
            self.order,
            self.params
        )
        self._getorder = dict([(i, Order(self, i)) for i in range(1, self.order+1)])

    @property
    def base(self):
        """Return base of logarithm tranform."""
        return self._gpm.base

    @property
    def logbase(self):
        """Return logarithmic function"""
        return self._gpm.logbase

    @property
    def n(self):
        """ Return the number of Interactions. """
        return len(self.labels)

    @property
    def log_transform(self):
        """ Boolean argument telling whether space is log transformed. """
        return self._gpm.log_transform

    @property
    def length(self):
        """ Length of sequences. """
        return self._gpm.length

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
    def params(self):
        """ Get the site-number-to-matrix-index mapping. This property is set in
        the build method.

        Returns
        -------
        params : dict
            { site_number : indices }`. If the site
            alphabet is note included, the model will assume binary
            between wildtype and derived.
            Example::
                mutations = {
                    0: [indices],
                    1: [indices],

                }
        """
        return self._params

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
    def n_replicates(self):
        """Get number of replicate measurements for observed phenotypes"""
        return self._gpm.n_replicates


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
        self.build()

    @labels.setter
    def labels(self, labels):
        """ Manually set the interactions considered in the map. Useful for building epistasis models manually. """
        self._labels = labels
        self._indices = np.arange(0, len(self.labels))

    @values.setter
    def values(self, values):
        """ Set the interactions of the system, set by an Epistasis model (see ..models.py)."""
        if len(values) != len(self.keys):
            raise Exception("Number of interactions give to map is different than was defined. ")
        self._values = values


    @keys.setter
    def keys(self, keys):
        """ Manually set keys. NEED TO do some quality control here. """
        self._keys = keys

    @stdeviations.setter
    def stdeviations(self, stdeviations):
        """Set the standard deviations of the epistatic coefficients."""
        self._stdeviations = stdeviations


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
        return np.array([i for i in range(len(labels)) if len(labels[i]) == self.order])

    @property
    def values(self):
        """Get values of epistasis for this order."""
        return self._epistasismap.values[self.indices]

    @property
    def keys(self):
        """Get keys of epistasis for this order."""
        return self._epistasismap.keys[self.indices]

    @property
    def stdeviations(self):
        """Get stdeviations of epistasis for this order."""
        return self._epistasismap.stdeviations[self.indices]
