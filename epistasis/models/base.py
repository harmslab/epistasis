import numpy as np
from functools import wraps

# imports from gpmap dependency
from gpmap.gpm import GenotypePhenotypeMap

# Local imports
from epistasis.mapping import EpistasisMap
from epistasis.decomposition import generate_dv_matrix
from epistasis.stats import bootstrap
from epistasis.utils import extract_mutations_from_genotypes

def sklearn_to_epistasis():
    """Decorate a subclass of sklearn model class to automagically convert it into a
    epistasis sklearn model class.
    """
    def inner(cls):
        base_model = cls.__bases__[-1]
        for attr in base_model.__dict__:
            method = getattr(base_model, attr)
            setattr(cls, attr, getattr(base_model, attr))
        return cls
    return inner

def X_predictor(method):
    """Decorator to automatically generate X for predictor methods in epistasis models."""
    @wraps(method)
    def inner(self, X=None):
        """"""
        # Build input to
        if X is None:
            X = self.X_constructor(genotypes=self.gpm.binary.complete_genotypes)
        return method(self, X)
    return inner

def X_fitter(method):
    """Decorator to automatically generate X for fit methods in epistasis models."""
    @wraps(method)
    def inner(self, X=None, y=None, **kwargs):
        # If no Y is given, try to get it from
        model_class = self.__class__.__name__
        if y is None:
            try:
                y = np.array(self.gpm.binary.phenotypes)
                if model_class in ["EpistasisLogisticRegression", "EpistasisSVC"]:
                    y[ y < self.threshold ] = 0
                    y[ y >= self.threshold ] = 1
            except AttributeError:
                raise AttributeError(
                    "y argument is missing, and no "
                    "GenotypePhenotypeMap is attached "
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
            output = method(self, X, y, **kwargs)
            # Reference the model coefficients in the epistasis map.
            self.epistasis.values = self.coef_
            return output
        else:
            self.X = X
            output = method(self, X, y, **kwargs)
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
        """ """
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
        setattr(self, "bootstrap_fit",self._bootstrap_fit)
        setattr(self, "bootstrap_predict",self._bootstrap_predict)
        setattr(self, "sample_fit",self._sample_fit)
        setattr(self, "sample_predict",self._sample_predict)

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
                self.epistasis = EpistasisMap.from_mutations(mutations, self.order, model_type=self.model_type)
        else:
            self.epistasis = EpistasisMap.from_labels(coeff_labels, model_type=self.model_type)
        # Construct the X matrix (convert to binary if necessary).
        try:
            return generate_dv_matrix(genotypes, self.epistasis.labels, model_type=self.model_type)
        except:
            mapping =self.gpm.map("complete_genotypes", "binary.complete_genotypes")
            binaries = [mapping[g] for g in genotypes]
            return generate_dv_matrix(binaries, self.epistasis.labels, model_type=self.model_type)

    def fit(self, *args, **kwargs):
        raise Exception("Must be defined in a subclass.")

    def predict(self, *args, **kwargs):
        raise Exception("Must be defined in a subclass.")

    def _sample_fit(self, *args, **kwargs):
        raise Exception("Must be defined in a subclass.")

    def _sample_predict(self, *args, **kwargs):
        raise Exception("Must be defined in a subclass.")

    @X_fitter
    def _bootstrap_fit(self, X=None, y=None, sample_weight=None, *args, **kwargs):
        """
        """
        mean, std, count = bootstrap(self.sample_fit, X=X, y=y, sample_weight=sample_weight, *args, **kwargs)
        return mean, std, count

    @X_predictor
    def _bootstrap_predict(self, X=None, *args, **kwargs):
        """
        """
        mean, std, count = bootstrap(self._sample_predict, X=X, *args, **kwargs)
        return mean, std, count
