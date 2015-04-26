import itertools as it
import numpy as np

from epistasis.mapping.base import BaseMap
from epistasis.mapping.mutation import MutationMap

class InteractionMap(BaseMap):
    
    def __init__(self):
        """ Mapping for interactions. """
        self.Mutations = MutationMap()
    
    @property
    def log_transform(self):
        """ Boolean argument telling whether space is log transformed. """
        return self._log_transform
    
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
    def errors(self):
        """ Get the value of the interaction errors in the system. """
        return self._errors
        
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
    def genotypes(self):
        """ Get the interaction genotype. """
        elements = ['w.t.']
        for label in self._labels[1:]:
            elements.append(self._label_to_genotype(label))
        return elements

    # ----------------------------------------------------------
    # Getter methods for mapping objects
    # ----------------------------------------------------------  
       
    @property
    def key2value(self):
        """ Return dict of interaction keys mapped to their values. """
        return OrderedDict([(self.keys[i], self.values[i]) for i in range(len(self.values))])
        
    @property
    def genotype2value(self):
        """ Return dict of interaction genotypes mapped to their values. """
        return OrderedDict([(self.genotypes[i], self.values[i]) for i in range(len(self.values))])
        return self._map(self.genotypes, self.values)
        
    @property
    def genotype2error(self):
        """ Return dict of interaction genotypes mapped to their values. """
        return OrderedDict([(self.genotypes[i], self.errors[:,i]) for i in range(len(self.values))])

    # ----------------------------------------------
    # Setter Functions
    # ----------------------------------------------
    @order.setter
    def order(self, order):
        """ Set the order of epistasis in the system. As a consequence, 
            this mapping object creates the """
        self._order = order
        # Create interaction labels and keys
        self.labels, self.keys, self._order_indices = self._build_interaction_map()
        self.indices = np.arange(len(self.labels))
        
    @values.setter
    def values(self, values):
        """ Set the interactions of the system, set by an Epistasis model (see ..models.py)."""
        if len(values) != len(self.labels):
            raise Exception("Number of interactions give to map is different than was defined. ")
        self._values = values
        
    @errors.setter
    def errors(self, errors):
        """ Set the interaction errors of the system, set by an Epistasis model (see ..models.py)."""
        if self.log_transform is True:
            if np.array(errors).shape != (2, len(self.labels)):
                raise Exception("""interaction_errors is not the right shape (should include 2 elements
                                    for each interaction, upper and lower bounds).""")
        else:
            if len(errors) != len(self.labels):    
                raise Exception("Number of interactions give to map is different than was defined. ")
        self._errors = errors

    @log_transform.setter
    def log_transform(self, boolean):
        """ True/False to log transform the space. """
        self._log_transform = boolean
        
    # ----------------------------------------------
    # Methods
    # ----------------------------------------------    

    def _build_interaction_map(self):
        """ Returns a label and key for every epistatic interaction. 
            
            Also returns a dictionary with order mapped to the index in the interactions array.
            
            An interaction label looks like [1,4,6] (type==list).
            An interaction key looks like '1,4,6'   (type==str).
        """
        labels = [[0]]
        keys = ['0']
        order_indices = dict()
        for o in range(1,self.order+1):
            start = len(labels)
            for label in it.combinations(range(1,self.Mutations.n+1), o):
                labels.append(list(label))
                key = ','.join([str(i) for i in label])
                keys.append(key)
            stop = len(labels)
            order_indices[o] = [start, stop]
        return labels, keys, order_indices

    def _label_to_genotype(self, label):
        """ Convert a label (list(3,4,5)) to its genotype representation ('A3V, A4V, A5V'). """
        genotype = ""
        for l in label:
            # Labels are offset by 1, remove offset for wildtype/mutation array index
            array_index = l - 1
            mutation = self.wildtype[array_index] + str(l) + self.Mutations.mutations[array_index]
            genotype += mutation + ','
        # Return genotype without the last comma
        return genotype[:-1]    