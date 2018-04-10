import json
import inspect
import numpy as np
import pandas as pd
from sklearn.preprocessing import binarize

from abc import abstractmethod, ABC, ABCMeta

# imports from gpmap dependency
from gpmap.gpm import GenotypePhenotypeMap

# Local imports
from epistasis.mapping import EpistasisMap, mutations_to_sites
from epistasis.matrix import get_model_matrix
from epistasis.utils import (extract_mutations_from_genotypes,
                             genotypes_to_X)
from .utils import XMatrixException
from sklearn.base import RegressorMixin, BaseEstimator

class SubclassException(Exception):
    """Subclass Exception for parent classes."""

def use_sklearn(sklearn_class):
    """Swap out base classes in an Epistasis model class with a sklearn_class +
    AbstractModel.
    """
    def mixer(cls):
        # Meta program the class
        name = cls.__name__
        methods = dict(cls.__dict__)

        # Put Sklearn first in line of parent classes
        parents = (sklearn_class, AbstractModel)

        # Rebuild class with Mixed in scikit learn.
        cls = type(name, parents, methods)
        return cls

    return mixer

class AbstractModel(ABC):
    """Abstract Base Class for all epistasis models.

    This class sets all docstrings not given in subclasses.
    """
    def __new__(self, *args, **kwargs):
        """Replace the docstrings of a subclass with docstrings in
        this base class.
        """
        # Get items in BaseModel.
        for name, member in inspect.getmembers(AbstractModel):
            # Get the docstring for this item
            doc = getattr(member, '__doc__')

            # Replace the docstring in self with basemodel docstring.
            try:
                member = getattr(self, name)
                member.__doc__ = doc

            except AttributeError:
                pass

        return super(AbstractModel, self).__new__(self)

    # --------------------------------------------------------------
    # Abstract Properties
    # --------------------------------------------------------------

    @property
    @abstractmethod
    def num_of_params(self):
        """Number of parameters in model.
        """
        raise SubclassException("Must be implemented in a subclass.")

    # --------------------------------------------------------------
    # Abstract Methods
    # --------------------------------------------------------------

    @abstractmethod
    def fit(self, X=None, y=None, **kwargs):
        """Fit model to data.

        Parameters
        ----------
        X : None, ndarray, or list of genotypes. (default=None)
            data used to construct X matrix that maps genotypes to
            model coefficients. If None, the model uses genotypes in the
            attached genotype-phenotype map. If a list of strings,
            the strings are genotypes that will be converted to an X matrix.
            If ndarray, the function assumes X is the X matrix used by the
            epistasis model.

        y : None or ndarray (default=None)
            array of phenotypes. If None, the phenotypes in the attached
            genotype-phenotype map is used.


        Returns
        -------
        self :
            The model is returned. Allows chaining methods.
        """
        raise SubclassException("Must be implemented in a subclass.")

    @abstractmethod
    def fit_transform(self, X=None, y=None, **kwargs):
        """Fit model to data and transform output according to model.

        Parameters
        ----------
        X : None, ndarray, or list of genotypes. (default=None)
            data used to construct X matrix that maps genotypes to
            model coefficients. If None, the model uses genotypes in the
            attached genotype-phenotype map. If a list of strings,
            the strings are genotypes that will be converted to an X matrix.
            If ndarray, the function assumes X is the X matrix used by the
            epistasis model.

        y : None or ndarray (default=None)
            array of phenotypes. If None, the phenotypes in the attached
            genotype-phenotype map is used.

        Returns
        -------
        gpm : GenotypePhenotypeMap
            The genotype-phenotype map object with transformed genotypes.
        """
        raise SubclassException("Must be implemented in a subclass.")

    @abstractmethod
    def predict(self, X=None):
        """Use model to predict phenotypes for a given list of genotypes.

        Parameters
        ----------
        X : None, ndarray, or list of genotypes. (default=None)
            data used to construct X matrix that maps genotypes to
            model coefficients. If None, the model uses genotypes in the
            attached genotype-phenotype map. If a list of strings,
            the strings are genotypes that will be converted to an X matrix.
            If ndarray, the function assumes X is the X matrix used by the
            epistasis model.

        Returns
        -------
        y : ndarray
            array of phenotypes.
        """
        raise SubclassException("Must be implemented in a subclass.")

    @abstractmethod
    def predict_transform(self, X=None, y=None, **kwargs):
        """Transform a set of phenotypes according to the model.

        Parameters
        ----------
        X : None, ndarray, or list of genotypes. (default=None)
            data used to construct X matrix that maps genotypes to
            model coefficients. If None, the model uses genotypes in the
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
        raise SubclassException("Must be implemented in a subclass.")

    @abstractmethod
    def hypothesis(self, X=None, thetas=None):
        """Compute phenotypes from given model parameters.

        Parameters
        ----------
        X : None, ndarray, or list of genotypes. (default=None)
            data used to construct X matrix that maps genotypes to
            model coefficients. If None, the model uses genotypes in the
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
        raise SubclassException("Must be implemented in a subclass.")

    @abstractmethod
    def hypothesis_transform(self, X=None, y=None, thetas=None):
        """Transform phenotypes with given model parameters.

        Parameters
        ----------
        X : None, ndarray, or list of genotypes. (default=None)
            data used to construct X matrix that maps genotypes to
            model coefficients. If None, the model uses genotypes in the
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
        raise SubclassException("Must be implemented in a subclass.")


    @abstractmethod
    def lnlike_of_data(
           self,
           X=None,
           y=None,
           yerr=None,
           thetas=None):
        """Compute the individUal log-likelihoods for each datapoint from a set
        of model parameters.

        Parameters
        ----------
        X : None, ndarray, or list of genotypes. (default=None)
            data used to construct X matrix that maps genotypes to
            model coefficients. If None, the model uses genotypes in the
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
        raise SubclassException("Must be implemented in a subclass.")

    @abstractmethod
    def lnlike_transform(
            self,
            X=None,
            y=None,
            yerr=None,
            lnprior=None,
            thetas=None):
        """Compute the individual log-likelihoods for each datapoint from a set
        of model parameters and a prior.

        Parameters
        ----------
        X : None, ndarray, or list of genotypes. (default=None)
            data used to construct X matrix that maps genotypes to
            model coefficients. If None, the model uses genotypes in the
            attached genotype-phenotype map. If a list of strings,
            the strings are genotypes that will be converted to an X matrix.
            If ndarray, the function assumes X is the X matrix used by the
            epistasis model.

        y : ndarray
            An array of phenotypes to transform.

        yerr : ndarray
            An array of the measured phenotypes standard deviations.

        lnprior : ndarray
            An array of priors for a given datapoint.

        thetas : ndarray
            array of model parameters. See thetas property for specifics.

        Returns
        -------
        y : ndarray
            array of phenotypes predicted by model parameters.
        """
        raise SubclassException("Must be implemented in a subclass.")

    def lnlikelihood(
            self,
            X=None,
            y=None,
            yerr=None,
            thetas=None):
        """Compute the individal log-likelihoods for each datapoint from a set
        of model parameters.

        Parameters
        ----------
        X : None, ndarray, or list of genotypes. (default=None)
            data used to construct X matrix that maps genotypes to
            model coefficients. If None, the model uses genotypes in the
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
        lnlike = np.sum(self.lnlike_of_data(X=X, y=y, yerr=yerr, thetas=thetas))

        # If log-likelihood is infinite, set to negative infinity.
        if np.isinf(lnlike) or np.isnan(lnlike):
            return -np.inf
        return lnlike

    def add_X(self, X=None, key=None):
        """Add X to Xbuilt

        Keyword arguments for X:

        - None :
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
        if isinstance(X, str) and X == None:

            if hasattr(self, "gpm") is False:
                raise XMatrixException("To build None, 'missing', or"
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
            raise XMatrixException("X must be one of the following: None, "
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

        # Map those columns to epistastalis dataframe.
        self.epistasis = EpistasisMap(
            sites=self.Xcolumns,
            order=self.order,
            model_type=self.model_type)
        return self

    @property
    def gpm(self):
        """Data stored in a GenotypePhenotypeMap object."""
        return self._gpm

    # -----------------------------------------------------------
    # Argument handlers.
    # -----------------------------------------------------------

    def _X(self, data=None, method=None):
        """Handle the X argument in this model."""
        X = data
        # If X is None, see if we saved an array.
        if X is None:
            # Get X from genotypes
            X = genotypes_to_X(
                self.gpm.wildtype,
                self.gpm.genotypes,
                order=self.order,
                mutations=self.gpm.mutations,
                model_type=self.model_type
            )

        elif isinstance(X, str) and X in self.gpm.genotypes:
            # Get X from genotypes
            X = genotypes_to_X(
                self.gpm.wildtype,
                [X],
                order=self.order,
                mutations=self.gpm.mutations,
                model_type=self.model_type
            )

        # If X is a keyword in Xbuilt, use it.
        elif isinstance(X, str) and X in self.Xbuilt:
            X = self.Xbuilt[X]

        # If 2-d array, keep as so.
        elif isinstance(X, np.ndarray) and X.ndim == 2:
            pass

        # If list of genotypes.
        elif isinstance(X, list) or isinstance(X, np.ndarray):
            # Get X from genotypes
            X = genotypes_to_X(
                self.gpm.wildtype,
                X,
                order=self.order,
                mutations=self.gpm.mutations,
                model_type=self.model_type
            )
        else:
            raise Exception("X is invalid.")

        # Save X
        self.Xbuilt[method] = X
        return X

    def _y(self, data=None, method=None):
        """Handle y arguments in this model."""
        y = data
        if y is None:
            return self.gpm.phenotypes

        elif isinstance(y, np.ndarray) or isinstance(y, list):
            return y

        else:
            raise Exception("y is invalid.")

    def _yerr(self, data=None, method=None):
        """Handle yerr argument in this model."""
        yerr = data
        if yerr is None:
            return self.gpm.std.upper

        elif isinstance(yerr, np.ndarray) or isinstance(yerr, list):
            return yerr
        else:
            raise Exception("yerr is invalid.")

    def _thetas(self, data=None, method=None):
        """Handle yerr argument in this model."""
        thetas = data
        if thetas is None:
            return self.thetas

        elif isinstance(thetas, np.ndarray) or isinstance(thetas, list):
            return thetas
        else:
            raise Exception("thetas is invalid.")

    def _lnprior(self, data=None, method=None):
        _lnprior = data
        if _lnprior is None:
            return np.zeros(self.gpm.n)

        elif isinstance(_lnprior, np.ndarray) or isinstance(_lnprior, list):
            return _lnprior
        else:
            raise Exceptison("_prior is invalid.")


class BaseModel(AbstractModel, RegressorMixin, BaseEstimator):
    """Base model for defining an epistasis model.
    """
    pass
