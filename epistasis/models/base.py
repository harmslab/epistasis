import numpy as np
from functools import wraps
from sklearn.preprocessing import binarize

# imports from gpmap dependency
from gpmap.gpm import GenotypePhenotypeMap

# Local imports
from epistasis.mapping import EpistasisMap
from epistasis.model_matrix_ext import get_model_matrix
from epistasis.utils import extract_mutations_from_genotypes

class BaseModel(object):
    """Base class for all models.

    Manages attachment of GenotypePhenotypeMap and EpistasisMaps to the Epistasis
    models.
    """
    @classmethod
    def from_json(cls, filename, **kwargs):
        """"""
        self = cls(**kwargs)
        self.add_gpm( GenotypePhenotypeMap.from_json(filename, **kwargs) )
        return self

    @classmethod
    def from_data(cls, wildtype, genotypes, phenotypes, **kwargs):
        """ Uses a simple linear, least-squares regression to estimate epistatic
        coefficients in a genotype-phenotype map. This assumes the map is linear."""
        self = cls(**kwargs)
        gpm = GenotypePhenotypeMap(wildtype, genotypes, phenotypes, **kwargs)
        self.add_gpm(gpm)
        return self

    @classmethod
    def from_gpm(cls, gpm, **kwargs):
        """ Initialize an epistasis model from a Genotype-phenotypeMap object """
        # Grab all properties from data-structure
        self = cls(**kwargs)
        self.add_gpm(gpm)
        return self

    def add_data(self, wildtype, genotypes, phenotypes, **kwargs):
        """Add genotype and phenotype data to the model.
        """
        # Build a genotype-phenotype map object from data.
        gpm = GenotypePhenotypeMap(wildtype, genotypes, phenotypes, **kwargs)
        self.add_gpm(gpm)
        return self

    def add_gpm(self, gpm):
        """Add a GenotypePhenotypeMap object to the epistasis model.

        Also exposes APIs that are only accessible with a GenotypePhenotypeMap
        attached to the model.
        """
        # Hacky way to
        instance_tree = (gpm.__class__,) + gpm.__class__.__bases__
        if GenotypePhenotypeMap in instance_tree is False:
            raise TypeError("gpm must be a GenotypePhenotypeMap object")
        self.gpm = gpm

    def fit(self, *args, **kwargs):
        raise Exception("Must be defined in a subclass.")

    def predict(self, *args, **kwargs):
        raise Exception("Must be defined in a subclass.")

    def hypothesis(self, *args, **kwargs):
        raise Exception("Must be defined in a subclass.")

    def lnlikelihood(self, *args, **kwargs):
        raise Exception("Must be defined in a subclass.")
