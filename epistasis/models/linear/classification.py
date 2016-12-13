from functools import wraps
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from ..base import BaseModel, X_fitter, X_predictor, sklearn_to_epistasis

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
        super(self.__class__, self).fit(X, y, sample_weight=None)

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

    @X_fitter
    def _sample_fit(self, X=None, y=None):
        """Sample the `fit` method from phenotype standard deviations."""
        # Fit a model
        y = self.gpm.stdeviations * np.random.randn() + self.gpm.phenotypes
        y[ y < self.threshold ] = 0
        y[ y >= self.threshold ] = 1
        model = self.__class__(
            threshold=self.threshold,
            order=self.order,
            model_type=self.model_type)
        model.fit(X=X, y=y)
        return model.coef_

    @X_fitter
    def _sample_predict(self, X=None, y=None):
        """Sample the `predict_proba` method from phenotype standard deviations.
        """
        # Fit a model
        y = self.gpm.stdeviations * np.random.randn() + self.gpm.phenotypes
        y[ y < self.threshold ] = 0
        y[ y >= self.threshold ] = 1
        model = self.__class__(
            threshold=self.threshold,
            order=self.order,
            model_type=self.model_type)
        model.fit(X=X, y=y)
        X_ = model.X_constructor(self.gpm.binary.complete_genotypes, mutations=self.gpm.mutations)
        # predict from that model
        predictions = model.predict_proba(X=X_)
        return predictions[:,0]


@sklearn_to_epistasis()
class EpistasisLogisticRegression(LogisticRegression, EpistasisBaseClassifier):
    """Logistic Regression used to categorize phenotypes as either alive or dead.
    """

@sklearn_to_epistasis()
class EpistasisSVC(SVC, EpistasisBaseClassifier):
    """Logistic Regression used to categorize phenotypes as either alive or dead.
    """
    @property
    def coef_(self):
        return self.support_vectors_
