import numpy as np
import pandas as pd

# Scikit-learn classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import binarize

from ..mapping import EpistasisMap
from .base import BaseModel, use_sklearn
from .utils import (XMatrixException, arghandler)

from .linear import EpistasisLinearRegression

from gpmap import GenotypePhenotypeMap

# Suppress Deprecation warning
import warnings
warnings.filterwarnings(action="ignore", module="sklearn",
                        category=DeprecationWarning)


@use_sklearn(LogisticRegression)
class EpistasisLogisticRegression(BaseModel):
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

    def fit(self, X=None, y=None, **kwargs):
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

    def fit_transform(self, X=None, y=None, **kwargs):
        self.fit(X=X, y=y, **kwargs)
        ypred = self.predict(X=X)


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

    @arghandler
    def _fit_(self, X=None, y=None, **kwargs):
        # Fit the classifier
        yclass = binarize(y.reshape(1, -1), self.threshold)[0]
        self.classes = yclass
        super(self.__class__, self).fit(X=X, y=yclass, **kwargs)
        self.epistasis.values = np.reshape(self.coef_, (-1,))
        return self

    @arghandler
    def predict(self, X=None):
        return super(self.__class__, self).predict(X=X)

    def predict_transform(self, X=None, y=None):
        x = self.predict(X=X)
        y[x <= 0.5] = self.threshold
        return y

    @arghandler
    def predict_log_proba(self, X=None):
        return super(self.__class__, self).predict_log_proba(X)

    @arghandler
    def predict_proba(self, X=None):
        return super(self.__class__, self).predict_proba(X)

    @arghandler
    def score(self, X=None, y=None, **kwargs):
        yclass = binarize(y.reshape(1, -1), threshold=self.threshold)[0]
        return super(self.__class__, self).score(X=X, y=yclass)

    @arghandler
    def lnlike_of_data(self, X=None, y=None, yerr=None, thetas=None):
        # Calculate Y's
        ymodel = self.hypothesis(X=X, thetas=thetas)
        ymodel_ = 1 - ymodel
        ymodel[ymodel < 0.5] = ymodel_[ymodel < 0.5]

        return np.log(ymodel)

    @arghandler
    def lnlike_transform(
            self,
            X=None,
            y=None,
            yerr=None,
            lnprior=None,
            thetas=None):
        # Update likelihood.
        ymodel = self.hypothesis(X=X, thetas=thetas)
        yclass = np.ones(len(ymodel))
        yclass[ymodel > 0.5] = 0

        lnlike = self.lnlike_of_data(X=X, y=y, yerr=yerr, thetas=thetas)
        lnprior[yclass == 0] = 0
        return lnlike + lnprior

    @arghandler
    def hypothesis(self, X=None, thetas=None):
        # Calculate probability of each class
        logit_p0 = 1 / (1 + np.exp(np.dot(X, thetas)))

        # Returns probability of class 1
        return logit_p0

    def hypothesis_transform(self, X=None, y=None, thetas=None):
        ypred = self.hypothesis(X=X, thetas=thetas)
        y[ypred > 0.5] = self.threshold
        return y

    @property
    def thetas(self):
        return self.epistasis.values
