
import json
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
    def __init__(self, order=1, *args, **kwargs):
        self.order=order
        self.Xbuilt = {}

    def fit(self, *args, **kwargs):
        raise Exception("Must be defined in a subclass.")

    def fit_transform(self, *args, **kwargs):
        raise Exception("Must be defined in a subclass.")

    def predict(self, *args, **kwargs):
        raise Exception("Must be defined in a subclass.")

    def predict_transform(self, *args, **kwargs):
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

    def add_X(self, X="complete", key=None):
        """Add X to Xbuilt

        Keyword arguments for X:

        - 'obs' :
            Uses ``gpm.binary`` to construct X. If genotypes
            are missing they will not be included in fit. At the end of
            fitting, an epistasis map attribute is attached to the model
            class.
        - 'missing' :
            Uses ``gpm.binary`` to construct X.
            All genotypes missing from the data are included. Warning,
            will break in most fitting methods. At the end of fitting,
            an epistasis map attribute is attached to the model class.
        - 'complete' :
            Uses ``gpm.binary`` to construct X.
            All genotypes missing from the data are included. Warning, will
            break in most fitting methods. At the end of fitting, an
            epistasis map attribute is attached to the model class.
        - 'fit' :
            a previously defined array/dataframe matrix. Prevents
            copying for efficiency.


        Parameters
        ----------
        X :
            see above for details.
        key : str
            name for storing the matrix.

        Returns
        -------
        Xbuilt : numpy.ndarray
            newly built 2d array matrix
        """
        if type(X) is str and X in ['obs', 'missing', 'complete', 'fit']:

            if hasattr(self, "gpm") is False:
                raise XMatrixException("To build 'obs', 'missing', or"
                                       "'complete' X matrix, a "
                                       "GenotypePhenotypeMap must be attached")

            # Get X columns
            columns = self.Xcolumns

            # Use desired set of genotypes for rows in X matrix.
            if X == "obs":
                index = self.gpm.binary
            elif X == "missing":
                index = self.gpm.missing_binary
            else:
                index = self.gpm.complete_binary

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

        Xbuilt = self.Xbuilt[key]
        return Xbuilt

    def add_gpm(self, gpm):
        """Add a GenotypePhenotypeMap object to the epistasis model.
        """
        # Hacky way to
        instance_tree = (gpm.__class__,) + gpm.__class__.__bases__
        if GenotypePhenotypeMap in instance_tree is False:
            raise TypeError("gpm must be a GenotypePhenotypeMap object")
        self._gpm = gpm

        # Construct columns for X matrix
        self.Xcolumns = mutations_to_sites(self.order, self.gpm.mutations)
        return self

    @property
    def gpm(self):
        """GenotypePhenotypeMap object"""
        return self._gpm

    @property
    def data(self):
        """Model data."""
        # Get dataframes
        df1 = self.gpm.complete_data
        df2 = self.epistasis.data

        # Merge dataframes.
        data = pd.concat((df1, df2), axis=1)
        return data

    def to_dict(self):
        """Return model data as dictionary."""
        # Get genotype-phenotype data
        data = self.data.to_dict(complete=True)

        # Update with model data
        data.update(model_type=self.model_type,
                    order=self.order)
        return data

    def to_excel(self, filename):
        """Write data to excel spreadsheet."""
        self.data.to_excel(filename)

    def to_csv(self, filename):
        """Write data to excel spreadsheet."""
        self.data.to_csv(filename)

    def to_json(self, filename):
        """Write to json file."""
        data = self.to_dict()
        with open(filename, 'w') as f:
            json.dump(data, f)
