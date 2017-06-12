from functools import wraps
import numpy as np

# Scikit-learn classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import binarize

from epistasis.models.base import BaseModel, X_fitter, X_predictor
from epistasis.models.utils import sklearn_to_epistasis

class EpistasisBaseClassifier(BaseModel):
    """Base class for implementing epistasis classification using scikit-learn models.
    To write your own epistasis classifier, write a subclass class, inherit whatever
    scikit-learn classifer class you'd like and this class (second).
    """
    def __init__(self, threshold, order=1, model_type="global", **kwargs):
        self.threshold = threshold
        self.order = order
        self.model_type = model_type
        super(self.__class__, self).__init__(**kwargs)

    @X_fitter
    def fit(self, X=None, y=None, sample_weight=None):
        # Build input linear regression.
        # Save the classes for y values.
        self.classes = binarize(y, self.threshold)[0]
        super(self.__class__, self).fit(X, y, sample_weight=None)
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
        return super(self.__class__, self).score(X, y)


@sklearn_to_epistasis()
class EpistasisLogisticRegression(LogisticRegression, EpistasisBaseClassifier):
    """Logistic Regression used to categorize phenotypes as either alive or dead."""

@sklearn_to_epistasis()
class EpistasisBernoulliNB(BernoulliNB, EpistasisBaseClassifier):
    """"""

@sklearn_to_epistasis()
class EpistasisSVC(SVC, EpistasisBaseClassifier):
    """Logistic Regression used to categorize phenotypes as either alive or dead."""
