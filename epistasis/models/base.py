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
                    n_replicates=1,
                    logbase=np.log10):

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
                        n_replicates=n_replicates,
                        logbase=logbase)

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
        # Grab all properties from data-structure
        options = {
            "log_transform": False,
            "mutations": None,
            "n_replicates": 1,
            "logbase": np.log10
        }

        # Get all options for map and order them
        for key in options:
            # See if options are in json data
            try:
                options[key] = getattr(gpm, key)
            except:
                pass

        options["stdeviations"] = stdeviations

        # Override any properties with specified kwargs passed directly into method
        options.update(kwargs)

        # Grab un scaled phenotypes and errors
        if gpm.log_transform is True:
            wildtype = gpm.wildtype
            genotypes = gpm.genotypes
            phenotypes = gpm.Raw.phenotypes
            stdeviations = gpm.Raw.stdeviations
        else:
            wildtype = gpm.wildtype
            genotypes = gpm.genotypes
            phenotypes = gpm.phenotypes
            stdeviations = gpm.stdeviations
            
        # Create an instance
        model = cls(args[0], args[1], args[2], **options)

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
