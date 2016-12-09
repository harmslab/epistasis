import numpy as np
from functools import wraps

# imports from seqspace dependency
from seqspace.gpm import GenotypePhenotypeMap

# Local imports
from epistasis.mapping import EpistasisMap
from epistasis.plotting.models import EpistasisPlotting
from epistasis.decomposition import generate_dv_matrix

def X_predictor(method):
    """Decorator to help automatically generate X for predictor methods in epistasis models."""
    @wraps(method)
    def inner(self, X=None, y=None):
        """"""
        # Build input to
        if X is None:
            X = self.X_constructor(genotypes=self.gpm.binary.complete_genotypes)
        return method(self, X)
    return inner

def X_fitter(method):
    """Decorator to help automatically generate X for fit methods in epistasis models."""
    @wraps(method)
    def inner(self, X=None, y=None, **kwargs):
        # If no Y is given, try to get it from
        model_class = self.__class__.__name__
        if y is None:
            try:
                y = np.array(self.gpm.binary.phenotypes)
                if model_class == "EpistasisLogisticRegression":
                    y[ y < self.threshold ] = 0
                    y[ y >= self.threshold ] = 1
            except AttributeError:
                raise AttributeError("y argument is missing, and no"
                                    "GenotypePhenotypeMap is attached"
                                    "to epistasis model.")
        # If X is not given, build one.
        if X is None:
            # See if an X already exists in the model
            try:
                X = self.X
            # If not, build one.
            except AttributeError:
                X = self.X_constructor(genotypes=self.gpm.genotypes)
                self.X = X
            output = method(self, X, y)
            # Reference the model coefficients in the epistasis map.
            self.epistasis.values = self.coef_
            return output
        else:
            output = method(self, X, y)
            return output
    return inner

class BaseModel(object):
    """ This object should be used as the parent class to all epistasis models.
    Manages attachment of GenotypePhenotypeMap and EpistasisMaps to the Epistasis
    models.
    """
    @classmethod
    def from_json(cls, filename, **kwargs):
        """"""
        self = cls(**kwargs)
        self.gpm = GenotypePhenotypeMap.from_json(filename)
        return self

    @classmethod
    def from_data(cls, wildtype, genotypes, phenotypes, **kwargs):
        """ Uses a simple linear, least-squares regression to estimate epistatic
        coefficients in a genotype-phenotype map. This assumes the map is linear."""

    @classmethod
    def from_gpm(cls, gpm, **kwargs):
        """ Initialize an epistasis model from a Genotype-phenotypeMap object """
        # Grab all properties from data-structure
        self = cls(**kwargs)
        self.gpm = gpm
        return self

    @classmethod
    def from_epistasis(cls, epistasis, **kwargs):
        """ """
        self = cls(**kwargs)
        self.epistasis = epistasis
        return self

    def attached_gpm(self, gpm):
        """ Attach a GenotypePhenotypeMap object to the epistasis model.
        """
        if gpm.__class__.__name__ != "GenotypePhenotypeMap":
            raise TypeError("gpm must be a GenotypePhenotypeMap object")
        self.gpm = gpm

    def X_constructor(self,
        genotypes=None,
        coeff_labels=None,
        mutations=None,
        **kwargs):
        """A helper method that constructs an X matrix for this model. Attaches
        an `EpistasisMap` object to the `epistasis` attribute of the model.

        The simplest way to construct X is to give a set of binary genotypes and
        epistatic labels. If not given, will try to infer these features from an
        attached genotype-phenotype map. If no genotype-phenotype map is attached,
        raises an exception.

        Parameters
        ----------
        genotypes : list
            list of genotypes.
        coeff_labels: list
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
        if coeff_labels is None:
            # See if an epistasis map was already created
            if hasattr(self, "epistasis") is False:
                # Mutations dictionary given? if not, try to infer one.
                if mutations is None:
                    try:
                        mutations = self.gpm.mutations
                    except AttributeError:
                        mutations = extract_mutations_from_genotypes(genotypes)
                # Construct epistasis mapping
                self.epistasis = EpistasisMap.from_mutations(mutations, self.order)
        else:
            self.epistasis = EpistasisMap.from_labels(coeff_labels)
        # Construct the X matrix (convert to binary if necessary).
        try:
            return generate_dv_matrix(genotypes, self.epistasis.labels, model_type=self.model_type)
        except:
            mapping =self.gpm.map("complete_genotypes", "binary.complete_genotypes")
            binaries = [mapping[g] for g in genotypes]
            return generate_dv_matrix(binaries, self.epistasis.labels, model_type=self.model_type)
