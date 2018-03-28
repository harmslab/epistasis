import abc
import json
import inspect
import numpy as np
import pandas as pd
from functools import wraps
from sklearn.preprocessing import binarize

# imports from gpmap dependency
from gpmap.gpm import GenotypePhenotypeMap

# Local imports
from epistasis.mapping import EpistasisMap, mutations_to_sites
from epistasis.model_matrix_ext import get_model_matrix
from epistasis.utils import extract_mutations_from_genotypes, DocstringMeta
from .utils import XMatrixException
from sklearn.base import RegressorMixin, BaseEstimator

def sklearn_mixin(sklearn_class):
    """Mixing a Scikit learn model."""
    def mixer(cls):
        # Meta program the class
        name = cls.__name__
        methods = cls.__dict__
        parents = cls.__bases__

        # Put Sklearn first in line of parent classes
        parents = (sklearn_class,) + parents

        # Rebuild class with Mixed in scikit learn.
        cls = type(name, parents, dict(methods))
        return cls
    return mixer

class BaseModel(abc.ABC, BaseEstimator, RegressorMixin):
    """Abstract Base Class for all epistasis models.
    """
    def __new__(self, *args, **kwargs):
        """Replace the docstrings of a subclass with docstrings in
        this base class.
        """
        # Get all attributes in the inherited class
        names = list(self.__dict__.keys())

        for name in names:
            item = getattr(self, name)

            # If this attr
            if not getattr(item, '__doc__'):
                try:
                    base_item = getattr(BaseModel, name)
                    item.__doc__ = base_item.__doc__
                except AttributeError:
                    pass

        return super(BaseModel, self).__new__(self, *args, **kwargs)


    # --------------------------------------------------------------
    # Abstract Properties
    # --------------------------------------------------------------

    @property
    @abc.abstractmethod
    def num_of_params(self):
        """Number of parameters in model.
        """
        pass

    # --------------------------------------------------------------
    # Abstract Methods
    # --------------------------------------------------------------

    @abc.abstractmethod
    def fit(self, X='obs', y='obs', **kwargs):
        """Fit model to data.

        Parameters
        ----------
        X : 'obs', ndarray, or list of genotypes. (default='obs')
            data used to construct X matrix that maps genotypes to
            model coefficients. If 'obs', the model uses genotypes in the
            attached genotype-phenotype map. If a list of strings,
            the strings are genotypes that will be converted to an X matrix.
            If ndarray, the function assumes X is the X matrix used by the
            epistasis model.

        y : 'obs' or ndarray (default='obs')
            array of phenotypes. If 'obs', the phenotypes in the attached
            genotype-phenotype map is used.


        Returns
        -------
        self :
            The model is returned. Allows chaining methods.
        """
        pass

    @abc.abstractmethod
    def fit_transform(self, X='obs', y='obs', **kwargs):
        """Fit model to data.

        Parameters
        ----------
        X : 'obs', ndarray, or list of genotypes. (default='obs')
            data used to construct X matrix that maps genotypes to
            model coefficients. If 'obs', the model uses genotypes in the
            attached genotype-phenotype map. If a list of strings,
            the strings are genotypes that will be converted to an X matrix.
            If ndarray, the function assumes X is the X matrix used by the
            epistasis model.

        y : 'obs' or ndarray (default='obs')
            array of phenotypes. If 'obs', the phenotypes in the attached
            genotype-phenotype map is used.

        Returns
        -------
        gpm : GenotypePhenotypeMap
            The genotype-phenotype map object with transformed genotypes.
        """
        pass

    @abc.abstractmethod
    def predict(self, X='obs'):
        """Use model to predict phenotypes for a given list of genotypes.

        Parameters
        ----------
        X : 'obs', ndarray, or list of genotypes. (default='obs')
            data used to construct X matrix that maps genotypes to
            model coefficients. If 'obs', the model uses genotypes in the
            attached genotype-phenotype map. If a list of strings,
            the strings are genotypes that will be converted to an X matrix.
            If ndarray, the function assumes X is the X matrix used by the
            epistasis model.

        Returns
        -------
        y : ndarray
            array of phenotypes.
        """
        pass

    @abc.abstractmethod
    def predict_transform(self, X='obs', y='obs', **kwargs):
        """Transform a set of phenotypes according to the model.

        Parameters
        ----------
        X : 'obs', ndarray, or list of genotypes. (default='obs')
            data used to construct X matrix that maps genotypes to
            model coefficients. If 'obs', the model uses genotypes in the
            attached genotype-phenotype map. If a list of strings,
            the strings are genotypes that will be converted to an X matrix.
            If ndarray, the function assumes X is the X matrix used by the
            epistasis model.

        y : ndarray
            An array of phenotypes to transform.

        Returns
        -------
        y_transform : ndarray
            array of phenotypes.
        """
        pass

    @abc.abstractmethod
    def hypothesis(self, X='obs', thetas=None):
        """Compute phenotypes from given model parameters.

        Parameters
        ----------
        X : 'obs', ndarray, or list of genotypes. (default='obs')
            data used to construct X matrix that maps genotypes to
            model coefficients. If 'obs', the model uses genotypes in the
            attached genotype-phenotype map. If a list of strings,
            the strings are genotypes that will be converted to an X matrix.
            If ndarray, the function assumes X is the X matrix used by the
            epistasis model.

        thetas : ndarray
            array of model parameters. See thetas property for specifics.

        Returns
        -------
        y : ndarray
            array of phenotypes predicted by model parameters.
        """
        pass

    @abc.abstractmethod
    def hypothesis_transform(self, X='obs', y='obs', thetas=None):
        """Transform phenotypes with given model parameters.

        Parameterss
        ----------
        X : 'obs', ndarray, or list of genotypes. (default='obs')
            data used to construct X matrix that maps genotypes to
            model coefficients. If 'obs', the model uses genotypes in the
            attached genotype-phenotype map. If a list of strings,
            the strings are genotypes that will be converted to an X matrix.
            If ndarray, the function assumes X is the X matrix used by the
            epistasis model.

        y : ndarray
            An array of phenotypes to transform.

        thetas : ndarray
            array of model parameters. See thetas property for specifics.

        Returns
        -------
        y : ndarray
            array of phenotypes predicted by model parameters.
        """
        pass

    @abc.abstractmethod
    def lnlike_of_data(
           self,
           X='obs',
           y='obs',
           yerr='obs',
           thetas=None):
        """Compute the individUal log-likelihoods for each datapoint from a set
        of model parameters.

        Parameters
        ----------
        X : 'obs', ndarray, or list of genotypes. (default='obs')
            data used to construct X matrix that maps genotypes to
            model coefficients. If 'obs', the model uses genotypes in the
            attached genotype-phenotype map. If a list of strings,
            the strings are genotypes that will be converted to an X matrix.
            If ndarray, the function assumes X is the X matrix used by the
            epistasis model.

        y : ndarray
            An array of phenotypes to transform.

        yerr : ndarray
            An array of the measured phenotypes standard deviations.

        thetas : ndarray
            array of model parameters. See thetas property for specifics.

        Returns
        -------
        y : ndarray
            array of phenotypes predicted by model parameters.
        """
        pass


    def lnlikelihood(
            self,
            X="obs",
            y="obs",
            yerr="obs",
            thetas=None):
        """Compute the individal log-likelihoods for each datapoint from a set
        of model parameters.

        Parameters
        ----------
        X : 'obs', ndarray, or list of genotypes. (default='obs')
            data used to construct X matrix that maps genotypes to
            model coefficients. If 'obs', the model uses genotypes in the
            attached genotype-phenotype map. If a list of strings,
            the strings are genotypes that will be converted to an X matrix.
            If ndarray, the function assumes X is the X matrix used by the
            epistasis model.

        y : ndarray
            An array of phenotypes to transform.

        yerr : ndarray
            An array of the measured phenotypes standard deviations.

        thetas : ndarray
            array of model parameters. See thetas property for specifics.

        Returns
        -------
        lnlike : float
            log-likelihood of the model parameters.
        """
        lnlike = np.sum(self.lnlike_of_data(X=X, y=y, yerr=yerr,
                                            sample_weight=sample_weight,
                                            thetas=thetas))

        # If log-likelihood is infinite, set to negative infinity.
        if np.isinf(lnlike) or np.isnan(lnlike):
            return -np.inf
        return lnlike

    def add_X(self, X="obs", key=None):
        """Add X to Xbuilt

        Keyword arguments for X:

        - 'obs' :
            Uses ``gpm.binary`` to construct X. If genotypes
            are missing they will not be included in fit. At the end of
            fitting, an epistasis map attribute is attached to the model
            class.


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
        if isinstance(X, str) and X == 'obs':

            if hasattr(self, "gpm") is False:
                raise XMatrixException("To build 'obs', 'missing', or"
                                       "'complete' X matrix, a "
                                       "GenotypePhenotypeMap must be attached")

            # Get X columns
            columns = self.Xcolumns

            # Use desired set of genotypes for rows in X matrix.
            index = self.gpm.binary

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
        self._gpm = gpm

        # Reset Xbuilt.
        self.Xbuilt = {}

        # Construct columns for X matrix
        self.Xcolumns = mutations_to_sites(self.order, self.gpm.mutations)
        return self

    @property
    def gpm(self):
        """Data stored in a GenotypePhenotypeMap object."""
        return self._gpm
