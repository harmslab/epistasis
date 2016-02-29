__doc__ = """
Base class for epistasis models. This is meant to be called in a subclass.
"""
import numpy as np
import itertools as it
from collections import OrderedDict

# imports from seqspace dependency
from seqspace.utils import farthest_genotype, binary_mutations_map

# Local imports
from epistasis.utils import epistatic_order_indices, SubclassException
from epistasis.mapping.epistasis import EpistasisMap

from epistasis.plotting import EpistasisPlotting

class BaseModel(EpistasisMap):
    
    def __init__(self, wildtype, genotypes, phenotypes, 
                    stdeviations=None, 
                    log_transform=False, 
                    mutations=None, 
                    n_replicates=1):
                    
        """ Populate an Epistasis mapping object. 
        
            __Arguments__:
            
            `wildtype` [str] : Wildtype genotype. Wildtype phenotype will be used as reference state.
            
            `genotypes` [array-like, dtype=str] : Genotypes in map. Can be binary strings, or not.
            
            `phenotypes` [array-like] : Quantitative phenotype values
            
            `stdevs` [array-like] : List of phenotype errors.
            
            `log_transform` [bool] : If True, log transform the phenotypes.
            
            `mutations` [dict]: Mapping dictionary for mutations at each site.
            
        """
        # Defaults to binary mapping if not specific mutations are named
        if mutations is None:
            mutant = farthest_genotype(wildtype, genotypes)
            mutations = binary_mutations_map(wildtype, mutant)
            
        super(BaseModel, self).__init__(wildtype, genotypes, phenotypes, 
                        stdeviations=stdeviations, 
                        log_transform=log_transform, 
                        mutations=mutations, 
                        n_replicates=n_replicates)
                        
        # Add plotting object if matplotlib is installed
        try:
            self.Plot = EpistasisPlotting(self)
        except Warning:
            pass

    # ---------------------------------------------------------------------------------
    # Loading method
    # ---------------------------------------------------------------------------------
        
    @classmethod    
    def from_gpm(cls, gpm, **kwargs):
        """ Initialize an epistasis model from a Genotype-phenotypeMap object """
        
        # Grab un scaled phenotypes and errors
        if gpm.log_transform is True:
            _phenotypes = gpm.Raw.phenotypes
            _stdeviations = gpm.Raw.stdeviations
        else:
            _phenotypes = gpm.phenotypes
            _stdeviations = gpm.stdeviations

        
        # Grab each property from map
        model = cls(gpm.wildtype, 
                    gpm.genotypes, 
                    _phenotypes,
                    stdeviations=_stdeviations,
                    mutations = gpm.mutations,
                    log_transform= gpm.log_transform,
                    n_replicates = gpm.n_replicates,
                    **kwargs)
        
        return model
        
    # ---------------------------------------------------------------------------------
    # Other methods
    # ---------------------------------------------------------------------------------        
            
    def fit(self):
        """ Fitting methods for epistasis models. """
        raise SubclassException("""Must be implemented in a subclass.""")
        
    def fit_error(self):
        """ Fitting method for errors in the epistatic parameters. """
        raise SubclassException("""Must be implemented in a subclass.""")
            
