import abc
from functools import wraps
import numpy as np
import pandas as pd

# Scikit-learn classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import binarize

from ..mapping import EpistasisMap
from .base import BaseModel, sklearn_mixin
from .utils import (XMatrixException,
                    X_fitter,
                    epistasis_fitter,
                    X_predictor)

from .linear import EpistasisLinearRegression

from gpmap import GenotypePhenotypeMap

# Suppress Deprecation warning
import warnings
warnings.filterwarnings(action="ignore", module="sklearn",
                        category=DeprecationWarning)


class ClassifierMixin(BaseModel):
    """Base class for implementing epistasis classification using scikit-learn
    models. To write your own epistasis classifier, write a subclass class,
    inherit whatever scikit-learn classifer class you'd like and this class
    (second).
    """
    def __init__(self, threshold, model_type="global", **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.threshold = threshold
        self.model_type = model_type
        self.fit_intercept = False
        self.order = 1
        self.Xbuilt = {}

        # Store model specs.
        self.model_specs = dict(
            threshold=self.threshold,
            model_type=self.model_type,
            **kwargs)

        # Set up additive linear model for pre-classifying
        self.Additive = EpistasisLinearRegression(
            order=1, model_type=self.model_type)

    def fit(self, X='obs', y='obs', **kwargs):
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
        self.fit(X=X, y=y, **kwargs)
        ypred = self.predict(X='fit')

        # Transform map.
        gpm = GenotypePhenotypeMap.read_dataframe(
            dataframe=self.gpm.data[ypred==1],
            wildtype=self.gpm.wildtype,
            mutations=self.gpm.mutations
        )
        return gpm

    @property
    def num_of_params(self):
        n = 0
        n += self.epistasis.n
        return n

    @epistasis_fitter
    @X_fitter
    def _fit_(self, X='obs', y='obs', **kwargs):
        # Fit the classifier
        yclass = binarize(y.reshape(1, -1), self.threshold)[0]
        self.classes = yclass
        super(self.__class__, self).fit(X=X, y=yclass, **kwargs)
        return self

    @X_predictor
    def predict(self, X='obs'):
        return super(self.__class__, self).predict(X)

    def predict_transform(self, X='obs', y='obs'):
        x = self.predict(X=X)

        if y is 'obs':
            y = self.gpm.phenotypes

        y[x == 0] = 0
        return y

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
        if thetas is None:
            thetas = self.thetas

        # Calculate Y's
        yclass = binarize(y.reshape(1, -1), threshold=self.threshold)[0]
        ymodel = self.hypothesis(X=X, thetas=thetas)

        # log-likelihood of logit model
        # NOTE: This likelihood is not normalized -- not a simple problem.
        return yclass * np.log(1 - ymodel) + (1 - yclass) * np.log(ymodel)


@sklearn_mixin(LogisticRegression)
class EpistasisLogisticRegression(ClassifierMixin):
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
        # Given thetas, estimate probability of class.
        if thetas is None:
            thetas = self.thetas

        # Calculate probability of each class
        logit_p1 = 1 / (1 + np.exp(np.dot(X, thetas)))

        # Determine class from probability
        classes = np.ones(len(logit_p1))
        classes[logit_p1 > 0.5] = 0

        # Return class
        return classes

    def hypothesis_transform(self, X='obs', y='obs'):
        pass

    @property
    def thetas(self):
        return self.epistasis.values

#
# @sklearn_mixin(BernoulliNB)
# class EpistasisBernoulliNB(ClassifierMixin):
#     """Naive Bayes Bernoulli Classifier."""
#
# @sklearn_mixin(SVC)
# class EpistasisSVC(ClassifierMixin):
#     """Support Vector Machine Classifier"""
