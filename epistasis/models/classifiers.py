from functools import wraps
import numpy as np
import pandas as pd

# Scikit-learn classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import binarize

from .base import BaseModel
from .utils import (sklearn_to_epistasis,
                    XMatrixException,
                    X_fitter,
                    X_predictor)

from .linear import EpistasisLinearRegression

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
        self.Additive.add_epistasis()

        # Fit the additive model and infer additive phenotypes
        self.Additive.fit(X=X, y=y)
        padd = self.Additive.predict(X=X)
        self = self._fit_(X=X, y=y)
        return self

    @X_fitter
    def _fit_(self, X='obs', y='obs', **kwargs):
        """"""
        # Fit the classifier
        yclass = binarize(y.values.reshape(1, -1), self.threshold)[0]
        self.classes = yclass
        super(self.__class__, self).fit(X=X, y=yclass, **kwargs)
        return self

    @X_predictor
    def predict(self, X='complete'):
        return super(self.__class__, self).predict(X)

    @X_predictor
    def predict_log_proba(self, X='complete'):
        return super(self.__class__, self).predict_log_proba(X)

    @X_predictor
    def predict_proba(self, X='complete'):
        return super(self.__class__, self).predict_proba(X)

    @X_fitter
    def score(self, X='obs', y='obs', **kwargs):
        yclass = binarize(y.values.reshape(1, -1), threshold=self.threshold)[0]
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
        yclass = binarize(y.values.reshape(1, -1), threshold=self.threshold)[0]
        ymodel = self.hypothesis(X=X, thetas=thetas)

        # log-likelihood of logit model
        # NOTE: This likelihood is not normalized -- not a simple problem.
        return yclass * np.log(1 - ymodel) + (1 - yclass) * np.log(ymodel)


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
