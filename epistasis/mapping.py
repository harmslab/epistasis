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

class TransformEpistasisMap(object):

    def __init__(self, EpistasisMap):
        self._epistasis = EpistasisMap
        self.transformed = True
        self.std = StandardDeviationMap(self)
        self.err = StandardErrorMap(self)

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

    @values.setter
    def values(self, values):
        """Set the non-log-epistasis."""


class EpistasisMap(BaseMap):

    def __init__(self, Model):
        """ Mapping object for indexing and tracking interactions in an
        epistasis map object.

        Parameters
        ----------
        Model :
            Epistasis Model to attach
        """
        self._Model = Model
        self.transformed = False
        if self._Model.log_transform:
            self.log = TransformEpistasisMap(self)
        self.std = StandardDeviationMap(self)
        self.err = StandardErrorMap(self)

    def build(self):
        """Build a mapping object for epistatic interactions."""
        # construct the mutations mapping
        self._params = params_index_map(self._Model.mutations)
        self._labels = build_model_params(
            self.length,
            self.order,
            self.params
        )

    @property
    def base(self):
        """Return base of logarithm tranform."""
        return self._Model.base

    @property
    def logbase(self):
        """Return logarithmic function"""
        return self._Model.logbase

    @property
    def n(self):
        """ Return the number of Interactions. """
        return len(self.labels)

    @property
    def log_transform(self):
        """ Boolean argument telling whether space is log transformed. """
        return self._Model.log_transform

    @property
    def length(self):
        """ Length of sequences. """
        return self._Model.length

    @property
    def order(self):
        """ Get order of epistasis in system. """
        return self._Model.order

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

    # ----------------------------------------------
    # Setter Functions
    # ----------------------------------------------

    @labels.setter
    def labels(self, labels):
        """ Manually set the interactions considered in the map. Useful for building epistasis models manually. """
        self._labels = labels
        self._indices = np.arange(0, len(self.labels))

    @values.setter
    def values(self, values):
        """ Set the interactions of the system, set by an Epistasis model (see ..models.py)."""
        if len(values) != len(self.labels):
            raise Exception("Number of interactions give to map is different than was defined. ")
        self._values = values

    @keys.setter
    def keys(self, keys):
        """ Manually set keys. NEED TO do some quality control here. """
        self._keys = keys

    # ----------------------------------------------
    # Methods
    # ----------------------------------------------

    def _label_to_genotype(self, label):
        """ Convert a label (list(3,4,5)) to its genotype representation ('A3V, A4V, A5V').

            NEED TO REFACTOR
        """
        genotype = ""
        for l in label:
            # Labels are offset by 1, remove offset for wildtype/mutation array index
            array_index = l - 1
            mutation = self.Mutations.wildtype[array_index] + str(self.Mutations.indices[l-1]+1) + self.Mutations.mutations[array_index]
            genotype += mutation + ','
        # Return genotype without the last comma
        return genotype[:-1]

    def get_order(self, order):
        """ Return a dictionary of interactions of a given order."""

        # Construct the set of model parameters for given order
        labels = build_model_params(self.length,
                            order,
                            self.mutations,
                            start_order=order)

        # Get a mapping of model labels to values
        key2value = self.map("keys", "values")

        # Built a dict of order interactions to values
        desired = {}
        for label in labels:
            key = label_to_key(label)
            desired[key] = key2value[key]

        return desired
