__doc__ = """
Base class for epistasis models. This is meant to be called in a subclass.
"""
import numpy as np
import itertools as it
from collections import OrderedDict

# imports from seqspace dependency
from seqspace.utils import farthest_genotype, binary_mutations_map

# Local imports
from epistasis.utils import epistatic_order_indices
from epistasis.mapping.epistasis import EpistasisMap

class BaseModel(EpistasisMap):
    
    def __init__(self, wildtype, genotypes, phenotypes, errors=None, log_transform=False, mutations=None):
        """ Populate an Epistasis mapping object. 
        
            __Arguments__:
            
            `wildtype` [str] : Wildtype genotype. Wildtype phenotype will be used as reference state.
            
            `genotypes` [array-like, dtype=str] : Genotypes in map. Can be binary strings, or not.
            
            `phenotypes` [array-like] : Quantitative phenotype values
            
            `errors` [array-like] : List of phenotype errors.
            
            `log_transform` [bool] : If True, log transform the phenotypes.
            
            `mutations` [dict]: Mapping dictionary for mutations at each site.
            
        """
        # Defaults to binary mapping if not specific mutations are named
        if mutations is None:
            mutant = farthest_genotype(wildtype, genotypes)
            mutations = binary_mutations_map(wildtype, mutant)
            
        super(BaseModel, self).__init__(wildtype, genotypes, phenotypes, errors=errors, log_transform=log_transform, mutations=mutations)
        
        # Construct a binary representation of the map (method inherited from parent class)
        # and make it a subclass of the model.
        self._construct_binary()
      
        # Model error if given. 
        if errors is not None:
            self.errors = errors
            
    # ---------------------------------------------------------------------------------
    # Loading method
    # ---------------------------------------------------------------------------------
        
    @classmethod    
    def from_gpm(cls, gpm, **kwargs):
        """ Initialize an epistasis model from a Genotype-phenotypeMap object """
        # Grab each property from map
        model = cls(gpm.wildtype, 
                    gpm.genotypes, 
                    gpm.Raw.phenotypes, 
                    errors = gpm.Raw.errors.upper,
                    mutations = gpm.mutations,
                    log_transform= gpm.log_transform,
                    **kwargs)
        
        return model
        
    # ---------------------------------------------------------------------------------
    # Other methods
    # ---------------------------------------------------------------------------------        
        
    def get_order(self, order, errors=False, label="genotype"):
        """ Return a dict of interactions to values of a given order. """
        
        # get starting index of interactions
        if order > self.order:
            raise Exception("Order argument is higher than model's order")
            
        # Determine the indices of this order of interactions.
        start, stop = epistatic_order_indices(self.length,order)
        # Label type.
        if label == "genotype":
            keys = self.Interactions.genotypes
        elif label == "keys":
            keys = self.Interactions.keys
        else:
            raise Exception("Unknown keyword argument for label.")
        
        # Build dictionary of interactions
        stuff = OrderedDict(zip(keys[start:stop], self.Interactions.values[start:stop]))
        if errors:
            errors = OrderedDict(zip(keys[start:stop], self.Interactions.errors[start:stop]))
            return stuff, errors
        else:
            return stuff
            
    def fit(self):
        """ Fitting methods for epistasis models. """
        raise Exception("""Must be implemented in a subclass.""")
        
    def fit_error(self):
        """ Fitting method for errors in the epistatic parameters. """
        raise Exception("""Must be implemented in a subclass.""")
            
