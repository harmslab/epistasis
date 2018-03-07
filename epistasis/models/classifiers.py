from functools import wraps
import numpy as np
import pandas as pd

# Scikit-learn classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import binarize

from ..mapping import EpistasisMap
from .base import BaseModel
from .utils import (sklearn_to_epistasis,
                    XMatrixException,
                    X_fitter,
                    epistasis_fitter,
                    X_predictor)

from .linear import EpistasisLinearRegression

from gpmap import GenotypePhenotypeMap

# Suppress Deprecation warning
import warnings
warnings.filterwarnings(action="ignore", module="sklearn",
                        category=DeprecationWarning)


class EpistasisBaseClassifier(BaseModel):
    """Base class for implementing epistasis classification using scikit-learn
    models. To write your own epistasis classifier, write a subclass class,
    inherit whatever scikit-learn classifer class you'd like and this class
    (second).
    """

    def __init__(self, threshold, order=1, model_type="global", **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.threshold = threshold
        self.order = order
        self.model_type = model_type
        self.fit_intercept = False
        self.Xbuilt = {}

        # Store model specs.
        self.model_specs = dict(
            order=self.order,
            threshold=self.threshold,
            model_type=self.model_type,
            **kwargs)

        # Set up additive linear model for pre-classifying
        self.Additive = EpistasisLinearRegression(
            order=1, model_type=self.model_type)

    def fit(self, X='obs', y='obs', **kwargs):
        """Fit Classifier to estimate the class of unknown phenotypes."""
        # Use Additive model to establish the phenotypic scale.
        # Prepare Additive model
        self.Additive.add_gpm(self.gpm)

        # Prepare a high-order model
        self.Additive.epistasis = EpistasisMap(
            sites=self.Additive.Xcolumns,
            order=self.Additive.order,
            model_type=self.Additive.model_type
        )

        # Fit the additive model and infer additive phenotypes
        self.Additive.fit(X=X, y=y)
        padd = self.Additive.predict(X=X)
        self = self._fit_(X=X, y=y)
        return self

    def fit_transform(self, X='obs', y='obs', **kwargs):
        """Fit and transform data for an Epistasis Pipeline.

        Returns
        -------
        gpm : GenotypePhenotypeMap
            data with phenotypes transformed according to model.
        """
        self.fit(X=X, y=y, **kwargs)
        ypred = self.predict(X='fit')
        yprob = self.predict_proba(X='fit')

        # Transform map.
        gpm = GenotypePhenotypeMap.read_dataframe(
            dataframe=self.gpm.data[ypred==1],
            wildtype=self.gpm.wildtype,
            mutations=self.gpm.mutations
        )
        return gpm

    @epistasis_fitter
    @X_fitter
    def _fit_(self, X='obs', y='obs', **kwargs):
        """Fit classifier."""
        # Fit the classifier
        yclass = binarize(y.reshape(1, -1), self.threshold)[0]
        self.classes = yclass
        super(self.__class__, self).fit(X=X, y=yclass, **kwargs)
        return self

    @X_predictor
    def predict(self, X='obs'):
        """Predict classes."""
        return super(self.__class__, self).predict(X)

    def predict_transform(self, X='obs', y='obs'):
        """Predict classes and apply to phenotypes. Used mostly in Pipeline
        object.
        """
        x = self.predict(X=X)

        if y is 'obs':
            return x * self.gpm.phenotypes
        else:
            return x * y

    @X_predictor
    def predict_log_proba(self, X='obs'):
        return super(self.__class__, self).predict_log_proba(X)

    @X_predictor
    def predict_proba(self, X='obs'):
        return super(self.__class__, self).predict_proba(X)

    @X_fitter
    def score(self, X='obs', y='obs', **kwargs):
        yclass = binarize(y.reshape(1, -1), threshold=self.threshold)[0]
        return super(self.__class__, self).score(X=X, y=yclass)

    @X_fitter
    def lnlike_of_data(self, X='obs', y='obs', yerr='obs',
                       sample_weight=None, thetas=None):
        """Calculate the log likelihoods of each data point, given a set of
        model coefficients.

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
        lnlike : np.ndarray
            log-likelihood of each data point given a model.
        """
        if thetas is None:
            thetas = self.thetas

        # Calculate Y's
        yclass = binarize(y.reshape(1, -1), threshold=self.threshold)[0]
        ymodel = self.hypothesis(X=X, thetas=thetas)

        # log-likelihood of logit model
        # NOTE: This likelihood is not normalized -- not a simple problem.
        return yclass * np.log(1 - ymodel) + (1 - yclass) * np.log(ymodel)

    @property
    def data(self):
        """Model data."""
        # Get dataframes
        df1 = self.gpm.complete_data
        df2 = self.Linear.epistasis.data

        # Merge dataframes.
        data = pd.concat((df1, df2), axis=1)
        return data

    @classmethod
    def read_json(cls, filename, **kwargs):
        """Read genotype-phenotype data from a json file."""
        with open(filename, 'r') as f:
            data = json.load(f)

        gpm = GenotypePhenotypeMap(wildtype=data['wildtype'],
                                   genotypes=data['genotypes'],
                                   phenotypes=data['phenotypes'],
                                   stdeviations=data['stdeviations'],
                                   mutations=data['mutations'],
                                   n_replicates=data['n_replicates'])

        additive_epistasis = EpistasisMap(sites=data['additive']['sites'],
                                          values=data['additive']['values'],
                                          model_type=model_type['model_type'])

        # Initialize a model
        self = cls(order=data['order'],
                   model_type=data['model_type'],
                   **kwargs)

        self.add_gpm(gpm)
        self.Additive.epistasis = epistasis
        return self

    def to_excel(self, filename):
        """Write data to excel spreadsheet."""
        self.data.to_excel(filename)

    def to_csv(self, filename):
        """Write data to excel spreadsheet."""
        self.data.to_csv(filename)

    def to_dict(self):
        """Return model data as dictionary."""
        # Get genotype-phenotype data
        data = self.gpm.to_dict(complete=True)

        # Update with epistasis model data
        data.update({'additive': self.Additive.epistasis.to_dict()})

        # Update with model data
        data.update(model_type=self.model_type,
                    order=self.order)
        return data

    def to_json(self, filename):
        """Write to json file."""
        data = self.to_dict()
        with open(filename, 'w') as f:
            json.dump(data, f)


@sklearn_to_epistasis()
class EpistasisLogisticRegression(LogisticRegression, EpistasisBaseClassifier):
    """Logistic regression for estimating epistatic interactions that lead to
    nonviable phenotypes. Useful for predicting viable/nonviable phenotypes.

    Parameters
    ----------
    threshold : float
        value below which phenotypes are considered nonviable.
    order : int
        order of epistasis model
    model_type : str (default="global")
        type of model matrix to use. "global" defines epistasis with respect to
        a background-averaged "genotype-phenotype". "local" defines epistasis
        with respect to the wildtype genotype.
    """
    @X_predictor
    def hypothesis(self, X='obs', thetas=None):
        """Returns the probability of the data given the model."""
        if thetas is None:
            thetas = self.thetas
        logit_p1 = 1 / (1 + np.exp(np.dot(X, thetas)))
        return logit_p1

    @property
    def thetas(self):
        return self.epistasis.values


@sklearn_to_epistasis()
class EpistasisBernoulliNB(BernoulliNB, EpistasisBaseClassifier):
    """"""


@sklearn_to_epistasis()
class EpistasisSVC(SVC, EpistasisBaseClassifier):
    """Logistic Regression used to categorize phenotypes as either alive
    or dead."""
