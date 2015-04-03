import numpy as np
import itertools as it
from collections import OrderedDict

# Decorator for error catching
def setting_error(func):
    """ Raise an AttributeError if _genotypes are not set before using any methods. """
    def wrapper(*args, **kwargs):
        try:
            output = func(*args, **kwargs)
            return output
        except AttributeError:
            raise AttributeError("'genotypes' property must be set before setting this attribute.")
    return wrapper

def hamming_distance(s1, s2):
    """ Return the Hamming distance between equal-length sequences """
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))


# -------------------------------------
# Main class for building epistasis map
# -------------------------------------

class EpistasisMap(object):
    """
        Object that maps epistasis in a genotype-phenotype map. 
        
        Attributes:
        ----------
        length: int, 
            length of genotypes
        n: int
            size of genotype-phenotype map
        order: int
            order of epistasis in system
        wildtype: str
            wildtype genotype
        mutations: array of chars
            individual mutations from wildtype that are in the system
        genotypes: array
            genotype in system
        phenotypes: array
            quantitative phenotypes in system
        phenotype_errors: array
            errors for each phenotype value
        indices: array
            genotype indices
        interactions: array
            epistatic interactions in the genotype-phenotype map
        interaction_error: array
            errors for each epistatic interaction
        interaction_indices: array
            indices for interaction's position in mutation matrix
        interaction_genotypes: array
            interactions as their mutation labels
        interaction_labels: list of lists
            List of interactions indices 
        interaction_keys: list
            List of interaction keys
    """
    # ---------------------
    # Getter methods
    # ---------------------
    
    @property
    def length(self):
        """ Get length of the genotypes. """
        return self._length    
    
    @property
    def n(self):
        """ Get number of genotypes, i.e. size of the system. """
        return self._n

    @property
    def order(self):
        """ Get order of epistasis in system. """
        return self._order
    
    @property
    def wildtype(self):
        """ Get reference genotypes for interactions. """
        return self._wildtype
        
    @property
    def mutations(self):
        """ Get possible that occur from reference system. """
        return self._mutations
    
    @property
    def genotypes(self):
        """ Get the genotypes of the system. """
        return self._genotypes
        
    @property
    def bits(self):
        """ Get Binary representation of genotypes. """
        return self._bits
        
    @property
    def bit_indices(self):
        """ Get indices of genotypes in self.genotypes that mapped to their binary representation. """
        return self._bit_indices
        
    @property
    def phenotypes(self):
        """ Get the phenotypes of the system. """
        return self._phenotypes
    
    @property
    def phenotype_errors(self):
        """ Get the phenotypes' errors in the system. """
        return self._phenotype_errors
    
    @property
    def indices(self):
        """ Return numpy array of genotypes position. """
        return self._indices
    
    @property
    def interactions(self):
        """ Get the values of the interaction in the system"""
        return self._interactions
        
    @property
    def interaction_errors(self):
        """ Get the value of the interaction errors in the system. """
        return self._interaction_errors
        
    @property
    def interaction_indices(self):
        """ Get the interaction index in interaction matrix. """
        return self._interaction_indices
        
    @property
    def interaction_genotypes(self):
        """ Get the interaction genotype. """
        return self._interaction_genotype
        
    @property
    def interaction_labels(self):
        """ Get the interaction labels, which describe the position of interacting mutations in
            the genotypes. (type==list of lists, see self._build_interaction_labels)
            """
        return self._interaction_labels
        
    @property
    def interaction_keys(self):
        """ Get the interaction keys. (type==list of str, see self._build_interaction_labels)"""
        return self._interaction_keys
        
    # ---------------------
    # Setter methods
    # ---------------------
    
    @genotypes.setter
    def genotypes(self, genotypes):
        """ Set genotypes from ordered list of sequences. """
        genotypes = sorted(genotypes)
        self._n = len(genotypes)
        self._length = len(genotypes[0])
        self._genotypes = np.array(genotypes)
        self._indices = np.arange(self._n)
        
    @wildtype.setter
    @setting_error
    def wildtype(self, wildtype):
        """ Set the reference genotype among the mutants in the system. """
        if type(wildtype) != str:
            raise Exception("Reference must be a string.")
        self._wildtype = wildtype
        self._mutations = self._differ_all_sites(wildtype)
        self._to_bits()
    
    @order.setter
    @setting_error
    def order(self, order):
        """ Set the order of epistasis in the system. As a consequence, 
            this mapping object creates the """
        self._order = order
        # Create interaction labels and keys
        self._interaction_labels, self._interaction_keys, self._order_indices = self._build_interaction_map()
        self._interaction_indices = np.arange(len(self._interaction_labels))
        
    @phenotypes.setter
    @setting_error
    def phenotypes(self, phenotypes):
        """ Set phenotypes from ordered list of phenotypes. 
            
            Args:
            -----
            phenotypes: array-like or dict
                if array-like, it musted be ordered by genotype; if dict,
                this method automatically orders the phenotypes into numpy
                array.
        """
        if type(phenotypes) is dict:
            self._phenotypes = self._if_dict(phenotypes)
        else:
            if len(phenotypes) != len(self._genotypes):
                raise("Number of phenotypes does not equal number of genotypes.")
            else:
                self._phenotypes = phenotypes
        
    @phenotype_errors.setter
    @setting_error
    def phenotype_errors(self, errors):
        """ Set error from ordered list of phenotype error. 
            
            Args:
            -----
            error: array-like or dict
                if array-like, it musted be ordered by genotype; if dict,
                this method automatically orders the errors into numpy
                array.
        """
        if type(errors) is dict:
            self._phenotype_errors = self._if_dict(phenotype_errors)
        else:
            self._phenotype_errors = errors
        
    @interactions.setter
    @setting_error
    def interactions(self, interactions):
        """ Set the interactions of the system."""
        self._interactions = interactions
        
    @interaction_errors.setter
    @setting_error
    def interaction_errors(self, interaction_errors):
        """ Set the interaction errors of the system."""
        self._interaction_errors = interaction_errors
        
    @interaction_indices.setter
    @setting_error
    def interaction_indices(self, interaction_indices):
        """ Set the genotypes of the system."""
        self._interaction_indices = interaction_indices
        
    @interaction_genotypes.setter
    @setting_error
    def interaction_genotypes(self, interaction_genotypes):
        """ Set the interaction genotypes of the system."""
        self._genotypes = genotypes
        
    @property
    def geno2binary(self):
        """ Return dictionary of genotypes mapped to their binary representation. """
        mapping = dict()
        for i in range(self.n):
            mapping[self.genotypes[self.bit_indices[i]]] = self.bits[i] 
        return mapping

    @property
    def geno2index(self):
        """ Return dict of genotypes mapped to their indices in transition matrix. """
        return self._map(self._genotypes, self._indices)
        
    
    # ---------------------------------
    # Useful methods for mapping object
    # ---------------------------------
        
    def _map(self, keys, values):
        """ Return ordered dictionary mapping two properties in self. """
        return OrderedDict([(keys[i], values[i]) for i in range(self._n)])
        
    def _if_dict(self, dictionary):
        """ If setter method is passed a dictionary with genotypes as keys, 
            use those keys to populate array of elements in order
        """
        elements = np.empty(self._n, dtype=float)
        for i in range(self._n):
            elements[i] = dictionary[self._genotypes[i]]
        return elements

    def _build_interaction_map(self):
        """ Returns a label and key for every epistatic interaction. 
            
            Also returns a dictionary with order mapped to the index in the interactions array.
            
            An interaction label looks like [1,4,6] (type==list).
            An interaction key looks like '1,4,6'   (type==str).
        """
        labels = [0]
        keys = ['0']
        order_indices = dict()
        for o in range(1,self._order+1):
            start = len(labels)
            for label in it.combinations(range(1,self._length+1), o):
                labels.append(list(label))
                key = ','.join([str(i) for i in label])
                keys.append(key)
            stop = len(labels)
            order_indices[o] = [start, stop]
        return labels, keys, order_indices
        
    def _differ_all_sites(self, reference):
        """ Find the genotype in the system that differs at all sites from reference.""" 
        for genotype in self.genotypes:
            if hamming_distance(genotype, reference) == self._length:
                differs = genotype
                break
        return genotype
        
    def _to_bits(self):
        """ Encode the genotypes an ordered binary set of genotypes with 
            wildtype as reference state (ref is all zeros).
        """
        w = list(self.wildtype)
        m = list(self.mutations)
        # build binary system
        binaries = sorted(["".join(list(s)) for s in it.product('01', repeat=len(w))])
        # get genotype indices
        geno2index = self.geno2index
        # initialize bit_indicies
        bit_indices = np.empty(len(binaries), dtype=int)
        for b in range(len(binaries)):
            binary = list(binaries[b])
            sequence = list()
            for i in range(len(w)):
                if binaries[b][i] == '0':
                    sequence.append(w[i])
                else:
                    sequence.append(m[i])
            # Find genotype in map and store index
            bit_indices[b] = geno2index[''.join(sequence)]
        self._bit_indices = bit_indices
        self._bits = binaries
            
                    
                    
    