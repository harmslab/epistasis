import numpy as _np
from sklearn.linear_model import LinearRegression as _LinearRegression

from .base import BaseModel as _BaseModel
from .utils import X_fitter as X_fitter
from .utils import X_predictor as X_predictor

# Suppress an annoying error from scikit-learn
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

class EpistasisLinearRegression(_LinearRegression, _BaseModel):
    """Ordinary least-squares regression for estimating high-order, epistatic
    interactions in a genotype-phenotype map.

    Methods are described in the following publication:
        Sailer, Z. R. & Harms, M. J. 'Detecting High-Order Epistasis in Nonlinear
        Genotype-Phenotype Maps'. Genetics 205, 1079-1088 (2017).

    Parameters
    ----------
    order : int
        order of epistasis
    model_type : str (default="global")
        model matrix type. See publication above for more information
    """
    def __init__(self, order=1, model_type="global", n_jobs=1, **kwargs):
        # Set Linear Regression settings.
        self.fit_intercept = False
        self.normalize = False
        self.copy_X = False
        self.n_jobs = n_jobs
        self.set_params(model_type=model_type, order=order)
        self.Xbuilt = {}
                
        # Store model specs.
        self.model_specs = dict(
            order=self.order,
            model_type=self.model_type,
            **kwargs)

    @X_fitter
    def fit(self, X='obs', y='obs', sample_weight=None, **kwargs):
        # If a threshold exists in the data, pre-classify genotypes
        return super(self.__class__, self).fit(X, y, sample_weight=sample_weight)

    @X_predictor
    def predict(self, X='complete'):
        return super(self.__class__, self).predict(X)

    @X_fitter
    def score(self, X='obs', y='obs', sample_weight=None):
        return super(self.__class__, self).score(X, y, sample_weight=None)

    @property
    def thetas(self):
        return self.coef_

    @X_predictor
    def hypothesis(self, X='complete', thetas=None):
        """Given a set of parameters, compute a set of phenotypes. This is method
        can be used to test a set of parameters (Useful for bayesian sampling).
        """
        if thetas is None:
            thetas = self.thetas
        return _np.dot(X, thetas)

    @X_fitter
    def lnlike_of_data(self, X="obs", y="obs", yerr="obs", thetas=None):
        """Calculate the log likelihoods of each data point, given a set of model coefficients.

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
        # If thetas are not explicitly named, get them from the model
        if thetas is None:
            thetas = self.thetas

        ######## Handle yerr.
        # Check if yerr is string
        if type(yerr) is str and yerr in ["obs", "complete"]:
            yerr = self.gpm.binary.std.upper

        # Else, numpy array or dataframe
        elif type(y) != np.array and type(y) != pd.Series:
            raise FittingError("yerr is not valid. Must be one of the following: 'obs', 'complete', "
                           "numpy.array, pandas.Series. Right now, its {}".format(type(yerr)))    

        # Calculate y from model.
        ymodel = self.hypothesis(X=X, thetas=thetas)
        return - 0.5 * _np.log(2*_np.pi*yerr**2) - 0.5*((y - ymodel)**2/yerr**2) 
