import numpy as np
import pandas as pd
from functools import wraps
from sklearn.preprocessing import binarize

# imports from gpmap dependency
from gpmap.gpm import GenotypePhenotypeMap

# Local imports
from epistasis.mapping import EpistasisMap, mutations_to_sites
from epistasis.model_matrix_ext import get_model_matrix
from epistasis.utils import extract_mutations_from_genotypes
from .utils import XMatrixException


class BaseModel(object):
    """Base class for all models.

    Manages attachment of GenotypePhenotypeMap and EpistasisMaps to the
    Epistasis models.
    """
    Xbuilt = {}

    def add_epistasis(self):
        """Add an EpistasisMap to model.
        """
        # Build epistasis interactions as columns in X matrix.
        sites = mutations_to_sites(self.order, self.gpm.mutations)

        # Map those columns to epistastalis dataframe.
        self.epistasis = EpistasisMap(
            sites, order=self.order, model_type=self.model_type)

    def add_X(self, X="complete", key=None):
        """Add X to Xbuilt

        X must be:

            - 'obs' : Uses `gpm.binary.genotypes` to construct X. If genotypes
                are missing they will not be included in fit. At the end of
                fitting, an epistasis map attribute is attached to the model
                class.
            - 'missing' : Uses `gpm.binary.missing_genotypes` to construct X.
                All genotypes missing from the data are included. Warning,
                will break in most fitting methods. At the end of fitting,
                an epistasis map attribute is attached to the model class.
            - 'complete' : Uses `gpm.binary.complete_genotypes` to construct X.
                All genotypes missing from the data are included. Warning, will
                break in most fitting methods. At the end of fitting, an
                epistasis map attribute is attached to the model class.
            - 'fit' : a previously defined array/dataframe matrix. Prevents
                copying for efficiency.

        Parameters
        ----------
        X :
            see above for details.
        key : str
            name for storing the matrix.

        Returns
        -------
        X_builts : numpy.ndarray
            newly built 2d array matrix
        """
        if type(X) is str and X in ['obs', 'missing', 'complete']:

            if hasattr(self, "gpm") is False:
                raise XMatrixException("To build 'obs', 'missing', or"
                                       "'complete' X matrix, a "
                                       "GenotypePhenotypeMap must be attached")

            # Create a list of epistatic interaction for this model.
            if hasattr(self, "epistasis"):
                columns = self.epistasis.sites
            else:
                self.add_epistasis()
                columns = self.epistasis.sites

            # Use desired set of genotypes for rows in X matrix.
            if X == "obs":
                index = self.gpm.binary.genotypes
            elif X == "missing":
                index = self.gpm.binary.missing_genotypes
            else:
                index = self.gpm.binary.complete_genotypes

            # Build numpy array
            x = get_model_matrix(index, columns, model_type=self.model_type)

            # Set matrix with given key.
            if key is None:
                key = X

            self.Xbuilt[key] = x

        elif type(X) == np.ndarray or type(X) == pd.DataFrame:
            # Set key
            if key is None:
                raise Exception("A key must be given to store.")

            # Store Xmatrix.
            self.Xbuilt[key] = X

        else:
            raise XMatrixException("X must be one of the following: 'obs', "
                                   "'complete', numpy.ndarray, or "
                                   "pandas.DataFrame.")

        X_built = self.Xbuilt[key]
        return X_built

    @classmethod
    def read_json(cls, filename, **kwargs):
        """Read genotype-phenotype data from a json file."""
        self = cls(**kwargs)
        self.add_gpm(GenotypePhenotypeMap.read_json(filename, **kwargs))
        return self

    @classmethod
    def read_excel(cls, filename, **kwargs):
        """Read genotype-phenotype data from a excel file."""
        self = cls(**kwargs)
        self.add_gpm(GenotypePhenotypeMap.read_excel(filename, **kwargs))
        return self

    @classmethod
    def read_csv(cls, filename, **kwargs):
        """Read genotype-phenotype data from a csv file."""
        self = cls(**kwargs)
        self.add_gpm(GenotypePhenotypeMap.read_csv(filename, **kwargs))
        return self

    @classmethod
    def read_data(cls, wildtype, genotypes, phenotypes, **kwargs):
        """ Uses a simple linear, least-squares regression to estimate epistatic
        coefficients in a genotype-phenotype map. This assumes the map is
        linear."""
        self = cls(**kwargs)
        gpm = GenotypePhenotypeMap(wildtype, genotypes, phenotypes, **kwargs)
        self.add_gpm(gpm)
        return self

    @classmethod
    def read_gpm(cls, gpm, **kwargs):
        """ Initialize an epistasis model from a Genotype-phenotypeMap object
        """
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

    def lnlike_of_data(self, *args, **kwargs):
        raise Exception("Must be defined in a subclass.")

    def lnlikelihood(self, X="obs", y="obs", yerr="obs",
                     sample_weight=None, thetas=None):
        """Calculate the log likelihood of y, given a set of model coefficients.

        Parameters
        ----------
        X : 2d array
            model matrix
        y : array
            data to calculate the likelihood
        yerr: array
            uncertainty in data
        thetas : array
            array of model coefficients

        Returns
        -------
        lnlike : float
            log-likelihood of data given a model.
        """
        lnlike = np.sum(self.lnlike_of_data(X=X, y=y, yerr=yerr,
                                            sample_weight=sample_weight,
                                            thetas=thetas))

        # If log-likelihood is infinite, set to negative infinity.
        if np.isinf(lnlike) or np.isnan(lnlike):
            return -np.inf
        return lnlike
