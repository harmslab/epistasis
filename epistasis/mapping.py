import numpy as np

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


class EpistasisMap(class):
    
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
    def genotypes(self):
        """ Get the genotypes of the system. """
        return self._genotypes
        
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
        """ Return numpy array of phenotypes. """
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
        return self._interaction_index
        
    @property
    def interaction_genotypes(self):
        """ Get the interaction genotype. """
        return self._interaction_genotype
        
    @property
    def interaction_labels(self):
        """ Get the interaction index in interaction matrixs. """
        return self._interaction_index
        
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
        
    @order.setter
    @setting_error
    def order(self, order):
        self._order = order
        
        
    @phenotypes.setter
    @setting_error
    def phenotypes(self, phenotypes):
        """ Set the phenotypes of the system."""
        self._phenotypes = phenotypes
        
    @phenotype_errors.setter
    @setting_error
    def phenotype_errors(self, phenotype_errors):
        """ Set the phenotype errors of the system."""
        self._phenotype_errors = phenotype_errors
        
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
        
    @interaction_labels.setter
    @setting_error
    def interaction_labels(self, interaction_labels):
        """ Set the interaction genotypes of the system."""
        self._interaction_labels = interaction_labels
        
    