# Suppress an annoying error
import warnings
# warnings.filterwarnings(action="ignore", category=RuntimeWarning)

# Scipy stack imports
import numpy as np
import pandas as pd

# Scikit learn imports
from sklearn.base import BaseEstimator, RegressorMixin

# GPMap import
from gpmap import GenotypePhenotypeMap

# Epistasis imports.
from epistasis.mapping import EpistasisMap
from epistasis.models.base import BaseModel
from epistasis.models.utils import (arghandler, FittingError)
from epistasis.models.linear import (EpistasisLinearRegression, EpistasisLasso)
from epistasis.stats import pearson

from .minimizer import FunctionMinimizer

class EpistasisNonlinearRegression(BaseModel):
    """Use nonlinear least-squares regression to estimate epistatic coefficients
    and nonlinear scale in a nonlinear genotype-phenotype map.

    This models has two steps:
        1. Fit an additive, linear regression to approximate the average effect
        of individual mutations.
        2. Fit the nonlinear function to the observed phenotypes vs. the
        additive phenotypes estimated in step 1.

    Methods are described in the following publication:
        Sailer, Z. R. & Harms, M. J. 'Detecting High-Order Epistasis in
        Nonlinear Genotype-Phenotype Maps'. Genetics 205, 1079-1088 (2017).

    Parameters
    ----------
    function : callable
        Nonlinear function between Pobs and Padd

    model_type : str (default: global)
        type of epistasis model to use. See paper above for more information.

    Keyword Arguments
    -----------------
    Keyword arguments are interpreted as intial guesses for the nonlinear
    function parameters. Must have the same name as parameters in the
    nonlinear function.

    Attributes
    ----------
    Additive : EpistasisLinearRegression
        Linear regression object for fitting additive model

    parameters : Parameters object
        Mapping object for nonlinear coefficients

    minimizer :
        Object that fits data using the function and a least squares minimization.
    """
    def __init__(self,
                 function,
                 model_type="global",
                 **p0):

        # Set up the function for fitting.
        self.function = function
        self.minimizer = FunctionMinimizer(self.function, **p0)
        self.parameters = self.minimizer.parameters
        self.order = 1
        self.Xbuilt = {}

        # Construct parameters object
        self.set_params(model_type=model_type)

        # Store model specs.
        self.model_specs = dict(
            function=self.function,
            model_type=self.model_type,
            **p0)

        # Set up additive and high-order linear model
        self.Additive = EpistasisLinearRegression(
            order=1, model_type=self.model_type)

    def add_gpm(self, gpm):
        super(EpistasisNonlinearRegression, self).add_gpm(gpm)
        # Add gpm to other models.
        self.Additive.add_gpm(gpm)
        return self

    @property
    def thetas(self):
        return np.concatenate((list(self.parameters.values()),
                               self.Additive.coef_))

    @property
    def num_of_params(self):
        n = 0
        n += len(self.parameters) + len(self.Additive.coef_)
        return n

    @arghandler
    def transform(self, X=None, y=None):
        # Use a first order matrix only.
        if type(X) == np.ndarray or type(X) == pd.DataFrame:
            Xadd = X[:, :self.Additive.epistasis.n]
        else:
            Xadd = X

        # Predict additive model.
        x = self.Additive.predict(X=Xadd)

        # Transform y onto x scale
        return self.minimizer.transform(x, y)

    def fit(self,
            X=None,
            y=None,
            **kwargs):
        # Fit linear portion
        self._fit_additive(X=X, y=y)

        # Step 2: fit nonlinear function
        self._fit_nonlinear(X=X, y=y, **kwargs)
        return self

    def _fit_additive(self, X=None, y=None, **kwargs):
        # Fit with an additive model
        self.Additive.epistasis = EpistasisMap(
            sites=self.Additive.Xcolumns,
            order=self.Additive.order,
            model_type=self.Additive.model_type
        )

        # Use a first order matrix only.
        if type(X) == np.ndarray or type(X) == pd.DataFrame:
            Xadd = X[:, :self.Additive.epistasis.n]
        else:
            Xadd = X

        # Fit Additive model
        self.Additive.fit(X=Xadd, y=y)
        self.Additive.epistasis.values = self.Additive.coef_
        return self

    @arghandler
    def _fit_nonlinear(self, X=None, y=None, **kwargs):
        """Estimate the scale of multiple mutations in a genotype-phenotype
        map."""
        # Use a first order matrix only.
        if type(X) == np.ndarray or type(X) == pd.DataFrame:
            Xadd = X[:, :self.Additive.epistasis.n]
        else:
            Xadd = X

        # Predict additive phenotypes.
        x = self.Additive.predict(X='fit')

        # Fit function
        self.minimizer.fit(x, y)
        self.parameters = self.minimizer.parameters

    @arghandler
    def fit_transform(self, X=None, y=None, **kwargs):
        self.fit(X=X, y=y, **kwargs)

        linear_phenotypes = self.transform(X=X, y=y)

        # Transform map.
        gpm = GenotypePhenotypeMap.read_dataframe(
            dataframe=self.gpm.data,
            wildtype=self.gpm.wildtype,
            mutations=self.gpm.mutations
        )

        gpm.data['phenotypes'] = linear_phenotypes
        return gpm

    def predict(self, X=None):
        x = self.Additive.predict(X=X)
        y = self.minimizer.predict(x)
        return y

    def predict_transform(self, X=None, y=None):
        if y is None:
            x = self.Additive.predict(X=X)
        else:
            x = y
        return self.minimizer.predict(x)

    @arghandler
    def hypothesis(self, X=None, thetas=None):
        # ----------------------------------------------------------------------
        # Part 0: Break up thetas
        # ----------------------------------------------------------------------
        i, j = len(self.parameters.valuesdict()), self.Additive.epistasis.n
        parameters = thetas[:i]
        epistasis = thetas[i:i + j]

        # Part 1: Linear portion
        x = np.dot(X, epistasis)

        # Part 2: Nonlinear portion
        ynonlin = self.minimizer.function(x, *parameters)

        return ynonlin

    def hypothesis_transform(self, X=None, y=None, thetas=None):
        # Break up thetas
        i, j = len(self.parameters.valuesdict()), self.Additive.epistasis.n
        parameters = thetas[:i]
        epistasis = thetas[i:i + j]

        if y is None:
            x = self.Additive.hypothesis(X=X, thetas=epistasis)
        else:
            x = y
        y_transform = self.minimizer.function(x, *parameters)
        return y_transform

    @arghandler
    def score(self, X=None, y=None):
        x = self.Additive.predict(X=X)
        ypred = self.minimizer.predict(x)
        return pearson(y, ypred)**2

    @arghandler
    def lnlike_of_data(self, X=None, y=None, yerr=None, thetas=None):
        # Calculate ymodel
        ymodel = self.hypothesis(X=X, thetas=thetas)

        # Likelihood of data given model
        return (- 0.5 * np.log(2 * np.pi * yerr**2) -
                (0.5 * ((y - ymodel)**2 / yerr**2)))

    @arghandler
    def lnlike_transform(
            self,
            X=None,
            y=None,
            yerr=None,
            lnprior=None,
            thetas=None):

        # Update likelihood.
        lnlike = self.lnlike_of_data(X=X, y=y, yerr=yerr, thetas=thetas)
        return lnlike + lnprior
