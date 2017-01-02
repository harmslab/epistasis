from functools import wraps
import numpy as np

from .base import BaseModel
from .linear.classification import EpistasisLogisticRegression
from .linear.regression import EpistasisLinearRegression
from .nonlinear.power import EpistasisPowerTransform

def gpm_check(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, "gpm") is False:
            raise Exception("A GenotypePhenotypeMap object must be attached")
        return method(self, *args, **kwargs)
    return wrapper

class EpistasisTwoStepRegression(BaseModel):
    """
    """
    def __init__(self, threshold, order=1, model_type="global", fix_linear=True, remove_zeros=True, **kwargs):
        self.remove_zeros = remove_zeros
        self.threshold = threshold
        self.order = order
        self.model_type = model_type
        self.fix_linear = fix_linear
        self.LogisticRegression = EpistasisLogisticRegression(threshold, order=order, model_type=model_type)
        self.PowerTransform = EpistasisPowerTransform(order=1, model_type=model_type, fix_linear=fix_linear)
        self.LinearRegression = EpistasisLinearRegression(order=order, model_type=model_type)

    @gpm_check
    def fit(self, y=None, **kwargs):
        """Fit data with classifier, scaling model, and linear high-order epistasis
        model.

        Parameters
        ----------
        X : array
            Training data
        y : array
            Target values
        sample_weight : array
            weights for each sample

        **kwargs must be arguments to nonlinear function.
        """
        # If no weights are given, determine weights from logistic regression.
        if y is None:
            y = self.gpm.phenotypes
        y_ = np.ones(self.gpm.n)
        y_[ y < self.threshold ] = 0
        X = self.X_constructor(self.gpm.binary.genotypes)
        X_ = self.PowerTransform.X_constructor(self.gpm.binary.genotypes)
        self.LogisticRegression.fit(X, y_)
        # Fit the logistic regression
        sample_weight = self.LogisticRegression.predict_proba(X)[:,1]

        if self.remove_zeros:
            subset = np.where( y_ == 1 )[0]
            y = y[subset]
            X = X[subset]
            X_ = X_[subset]
            sample_weight = sample_weight[subset]

        # Fit a nonlinear model.
        self.PowerTransform.fit(X=X_, y=y, sample_weight=sample_weight, **kwargs)
        self.parameters = self.PowerTransform.parameters

        # Fit linear epistasis model
        ylin = self.PowerTransform.transform_target(y)
        self.LinearRegression.fit(X, ylin, sample_weight=sample_weight)
        self.epistasis.values = self.LinearRegression.coef_

    @gpm_check
    def predict(self, X=None):
        """Predict targets from multiple model regressors.

        Parameters
        ----------
        X : ndarray
            Matrix to predict from regressors.

        Returns
        -------
        y_predicted : array
            Returns predicted values.
        """
        if X is None:
            X = self.LinearRegression.X_constructor(self.gpm.binary.complete_genotypes)
        y_linear = self.LinearRegression.predict(X)
        y_nonlinear = self.PowerTransform.function(y_linear, *self.parameters.get_params())
        y_predicted = y_nonlinear * self.LogisticRegression.predict(X)
        y_predicted = np.nan_to_num(y_predicted)
        return y_predicted

    def _sample_fit(self, n_samples=1, **kwargs):
        """"""
        raise Exception("""This method is not available for Two Part Regression.""")

    def _sample_predict(self, n_samples=1, min_score=0.0, **kwargs):
        """"""
        model = self.__class__(
            self.threshold,
            order=self.order,
            model_type=self.model_type,
            fix_linear=self.fix_linear,
            remove_zeros=self.remove_zeros
        )
        model.attach_gpm(self.gpm)
        X_ = model.X_constructor(self.gpm.binary.complete_genotypes)
        predictions = np.empty((len(self.gpm.complete_genotypes), n_samples), dtype=float)
        scores = np.empty(n_samples, dtype=float)
        count, failed_attempts = 0, 0
        while count < n_samples or failed_attempts > 1000:
            y = self.gpm.err.upper * np.random.randn(self.gpm.n) + self.gpm.phenotypes
            model.fit(y=y, **kwargs)
            y_target = model.predict(X_)
            score = model.PowerTransform._score
            if score < min_score:
                failed_attempts += 1
            else:
                scores[count] = score
                predictions[:, count] = y_target
                count += 1
        if failed_attempts == 1000:
            raise Exception("Failed to find samples. Lower your min_score.")
        return predictions, scores
