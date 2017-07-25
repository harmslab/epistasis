from functools import wraps
import numpy as np

# Scikit-learn classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import binarize

from .base import BaseModel
from .utils import sklearn_to_epistasis, X_fitter, X_predictor

# Suppress Deprecation warning
import warnings
warnings.filterwarnings(action="ignore", module="sklearn", category=DeprecationWarning)

class EpistasisBaseClassifier(BaseModel):
    """Base class for implementing epistasis classification using scikit-learn models.
    To write your own epistasis classifier, write a subclass class, inherit whatever
    scikit-learn classifer class you'd like and this class (second).
    """
    def __init__(self, threshold, order=1, model_type="global", **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.threshold = threshold
        self.order = order
        self.model_type = model_type
        self.fit_intercept=False

    @X_fitter
    def fit(self, X=None, y=None, **kwargs):
        # Save the classes for y values.
        self.classes = binarize(y, self.threshold)[0]
        super(self.__class__, self).fit(X, y=self.classes, **kwargs)
        return self

    @X_predictor
    def predict(self, X=None):
        return super(self.__class__, self).predict(X)

    @X_predictor
    def predict_log_proba(self, X=None):
        return super(self.__class__, self).predict_log_proba(X)

    @X_predictor
    def predict_proba(self, X=None):
        return super(self.__class__, self).predict_proba(X)

    @X_fitter
    def score(self, X=None, y=None):
        y = binarize(y, self.threshold)[0]
        return super(self.__class__, self).score(X, y)

    def lnlikelihood(self, X=None, ydata=None, thetas=None):
        """Calculate the log likelihood of data, given a set of model coefficients.

        Parameters
        ----------
        X : 2d array
            model matrix
        ydata : array
            data to calculate the likelihood
        yerr: array
            uncertainty in data
        thetas : array
            array of model coefficients

        Returns
        -------
        lnlike : float
            log-likelihood of the data given the model.
        """
        # 1. Class probability given the coefs
        if ymodel is None:
            ydata = self.gpm.phenotypes
        if X is None:
            X = self.Xfit
        ydata = binarize(ydata, threshold=self.threshold)[0]
        ymodel = self.hypothesis(X=X, thetas=thetas)
        ### log-likelihood of logit model
        return ydata * np.log(ymodel) + (1 - ydata) * np.log(1-ymodel)

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
    def hypothesis(self, X=None, thetas=None):
        """Returns the probability of the data given the model."""
        if thetas is None:
            thetas = self.thetas
        logit_p1 = 1 - 1 / (1 + np.exp(np.dot(X, thetas)))
        return logit_p1

    @property
    def thetas(self):
        return self.epistasis.values


@sklearn_to_epistasis()
class EpistasisBernoulliNB(BernoulliNB, EpistasisBaseClassifier):
    """"""

@sklearn_to_epistasis()
class EpistasisSVC(SVC, EpistasisBaseClassifier):
    """Logistic Regression used to categorize phenotypes as either alive or dead."""


@sklearn_to_epistasis()
class EpistasisGaussianProcessClassifier(GaussianProcessClassifier, EpistasisBaseClassifier):
    """"""
