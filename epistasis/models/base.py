import numpy as np
from functools import wraps
from sklearn.preprocessing import binarize

# imports from gpmap dependency
from gpmap.gpm import GenotypePhenotypeMap

# Local imports
from .utils import X_predictor, X_fitter
from epistasis.mapping import EpistasisMap
from epistasis.decomposition import generate_dv_matrix
from epistasis.utils import extract_mutations_from_genotypes

class BaseModel(object):
    """ This object should be used as the parent class to all epistasis models.
    Manages attachment of GenotypePhenotypeMap and EpistasisMaps to the Epistasis
    models.
    """
    @classmethod
    def from_json(cls, filename, **kwargs):
        """"""
        self = cls(**kwargs)
        self.attach_gpm( GenotypePhenotypeMap.from_json(filename, **kwargs) )
        return self

    @classmethod
    def from_data(cls, wildtype, genotypes, phenotypes, **kwargs):
        """ Uses a simple linear, least-squares regression to estimate epistatic
        coefficients in a genotype-phenotype map. This assumes the map is linear."""
        self = cls(**kwargs)
        gpm = GenotypePhenotypeMap(wildtype, genotypes, phenotypes, **kwargs)
        self.attach_gpm(gpm)
        return self

    @classmethod
    def from_gpm(cls, gpm, **kwargs):
        """ Initialize an epistasis model from a Genotype-phenotypeMap object """
        # Grab all properties from data-structure
        self = cls(**kwargs)
        self.attach_gpm(gpm)
        return self

    @classmethod
    def from_epistasis(cls, epistasis, **kwargs):
        """"""
        self = cls(**kwargs)
        self.epistasis = epistasis
        return self

    def attach_gpm(self, gpm):
        """ Attach a GenotypePhenotypeMap object to the epistasis model.

        Also exposes APIs that are only accessible with a GenotypePhenotypeMap
        attached to the model.
        """
        # Hacky way to
        instance_tree = (gpm.__class__,) + gpm.__class__.__bases__
        if GenotypePhenotypeMap in instance_tree is False:
            raise TypeError("gpm must be a GenotypePhenotypeMap object")
        self.gpm = gpm

    def X_constructor(self,
        genotypes=None,
        coeff_sites=None,
        mutations=None,
        **kwargs):
        """A helper method that constructs linear decomposition matrix, X, for
        an epistasis model. Attaches an `EpistasisMap` object to the `epistasis`
        attribute of the model to allow simple access to X coefficients.

        The simplest way to construct X is to give a set of binary genotypes and
        epistatic sites. If not given, will try to infer these features from an
        attached genotype-phenotype map. If no genotype-phenotype map is attached,
        raises an exception.

        Parameters
        ----------
        genotypes : list
            list of genotypes.
        coeff_sites: list
            list of lists. Each sublist contains site-indices that represent
            participants in that epistatic interaction.
        mutations : dict
            mutations dictionary mapping sites to alphabet at the site.
        """
        # First check genotypes are available
        if genotypes is None:
            try:
                genotypes = self.gpm.binary.genotypes
            except AttributeError:
                raise AttributeError("genotypes must be given, because no GenotypePhenotypeMap is attached to this model.")
        # Build epistasis map
        if coeff_sites is None:
            # See if an epistasis map was already created
            if hasattr(self, "epistasis") is False:
                # Mutations dictionary given? if not, try to infer one.
                if mutations is None:
                    try:
                        mutations = self.gpm.mutations
                    except AttributeError:
                        mutations = extract_mutations_from_genotypes(genotypes)
                # Construct epistasis mapping
                self.epistasis = EpistasisMap.from_mutations(mutations, self.order, model_type=self.model_type)
        else:
            self.epistasis = EpistasisMap.from_sites(coeff_sites, model_type=self.model_type)
        # Construct the X matrix (convert to binary if necessary).
        try:
            return generate_dv_matrix(genotypes, self.epistasis.sites, model_type=self.model_type)
        except:
            mapping = self.gpm.map("complete_genotypes", "binary.complete_genotypes")
            binaries = [mapping[g] for g in genotypes]
            return generate_dv_matrix(binaries, self.epistasis.sites, model_type=self.model_type)

    def fit(self, *args, **kwargs):
        raise Exception("Must be defined in a subclass.")

    def predict(self, *args, **kwargs):
        raise Exception("Must be defined in a subclass.")

    def _sample_fit(self, *args, **kwargs):
        raise Exception("Must be defined in a subclass.")

    def _sample_predict(self, *args, **kwargs):
        raise Exception("Must be defined in a subclass.")
