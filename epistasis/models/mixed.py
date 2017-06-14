from sklearn.preprocessing import binarize

from .nonlinear import EpistasisNonlinearRegression
from .classifiers import EpistasisLogisticRegression
from .utils import X_fitter, X_predictor


class EpistasisMixedLinearRegression(EpistasisNonlinearRegression):
    """
    """
    def __init__(self, threhold, **kwargs):
        self.threshold = threshold
        super(EpistasisMixedRegression, self).__init__(**kwargs)

    def binarize(self, y=None):
        """Transform y, random continuous variables, to binary variables around
        the model's threshold. If no y is given, the GenotypePhenotypeMap phenotypes
        will be used.
        """
        if y is None:
            y = self.gpm.phenotypes
        return binarize(y, self.threshold)[0]

    @X_fitter
    def fit(self, X=None, y=None, **kwargs):
        ybin = binarize(y=y)
        self.Classifer = EpistasisLogisticRegression.from_gpm(self.gpm,
            model_type=self.model_type,
            order=1).fit()

        super(EpistasisMixedRegression, self).fit(X=X, y=ybin, **kwargs)

    @X_predictor
    def predict(self):
        pass

    @X_predictor
    def hypothesis(self, X=None, thetas):
        """
        """
        logit_p = 1 / (1 + np.exp(-np.dot(X, thetas))
        return logit_p

    @property
    def thetas(self):
        pass
