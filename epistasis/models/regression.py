# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import numpy as np
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------
# Local imports
# ------------------------------------------------------------

from epistasis.decomposition import generate_dv_matrix
from epistasis.models.base import BaseModel
from epistasis.plotting.regression import RegressionPlotting
from epistasis.stats import resample_to_convergence
from epistasis.utils import (epistatic_order_indices,
                                build_model_params)


# ------------------------------------------------------------
# Unique Epistasis Functions
# ------------------------------------------------------------

class RegressionStatistics(object):
    """

    """
    def __init__(self, model):
        self.model = model
        self.score = None
        self.significance = None

class LinearEpistasisRegression(BaseModel):
    """ Uses a simple linear, least-squares regression to estimate epistatic
    coefficients in a genotype-phenotype map. This assumes the map is linear.

    Parameters
    ----------
    wildtype : str
        Wildtype genotype. Wildtype phenotype will be used as reference state.
    genotypes : array-like, dtype=str
        Genotypes in map. Can be binary strings, or not.
    phenotypes : array-like
        Quantitative phenotype values
    order : int
        Order of regression; if None, parameters must be passed in manually as parameters=<list of lists>
    parameters : dict
        interaction keys with their values expressed as lists.
    errors : array-like
        List of phenotype errors.
    log_transform : bool
        If True, log transform the phenotypes.
    mutations : dict
        site-by-site mutation scheme.
    model : str, default='local'
        either 'global' or 'local'. If 'local', wildtype is reference state. If 'global',
        is cleverly chosen average state.
    """
    def __init__(self, wildtype, genotypes, phenotypes,
        order=None,
        stdeviations=None,
        log_transform=False,
        mutations=None,
        n_replicates=1,
        model_type="local",
        logbase=np.log10):
        # Populate Epistasis Map
        super(LinearEpistasisRegression, self).__init__(wildtype, genotypes, phenotypes,
                stdeviations=stdeviations,
                log_transform=log_transform,
                mutations=mutations,
                n_replicates=n_replicates,
                logbase=logbase)

        # Generate basis matrix for mutant cycle approach to epistasis.
        if order is not None:
            self.order = order
        else:
            self.order = self.binary.length

        # Construct the epistasis map
        self.epistasis.order = self.order

        # Define the encoding for different models
        model_types = {
            "local": {
                "encoding": {"1": 1, "0": 0},       # Decomposition matrix encoding
            },
            "global": {
                "encoding": {"1": 1, "0": -1},
            }
        }

        self.model_type = model_type

        # Set encoding from model_type given
        self.encoding = model_types[model_type]["encoding"]
        # Construct decomposition matrix
        self.X = self.build_X(self.binary.genotypes, self.epistasis.labels, encoding=self.encoding)

        # Initialize useful objects to model object
        self.statistics = RegressionStats(self)

        # Try to add plotting module if matplotlib is installed
        try:
            self.plot = RegressionPlotting(self)
        except Warning:
            pass

    def build_X(self, genotypes):
        """ Construct a model matrix for linear regression.

        Parameters
        ----------
        genotypes : list
            list of genotypes, in binary representation. if not in binary format
            already, this method will try to do the conversion itself.

        Returns
        -------
        X : 2d array
            model matrix from this model to be used for linear regression.
        """
        # Make sure `genotypes` is a list object
        if len(genotypes) is 1:
            genotypes = [genotypes]
        else:
            genotypes = list(genotypes)
        # Construct the X matrix (convert to binary if necessary).
        try:
            return generate_dv_matrix(genotypes, self.epistasis.labels, encoding=self.encoding)
        except:
            mapping =self.map("genotypes", "binaries")
            binaries = [mapping[g] for g in genotypes]
            return generate_dv_matrix(binaries, self.epistasis.labels, encoding=self.encoding)

    def predict(self, X=None):
        """ Predict phenotypes from linear epistasis model.

        Parameters
        ----------
        X : 2d array (optional)
            X matrix passed to scikit-learn's `LinearRegression` class. If no
            matrix is given, will create a matrix from the `complete_genotypes`
            attribute

        Returns
        -------
        phenotypes : array
            phenotypes predicted from linear regression model.
        """
        if X is None:
            X = self.build_X(self.binary.complete_genotypes)
        phenotypes = self.regression_model.predict(X)
        # If a log transform was used, reverse-transform the phenotypes.
        if self.log_transform:
            phenotypes = self.base**phenotypes
        return phenotypes

    def fit(self):
        """Use ordinary least squares regression (via scikit-learn) to estimate
        the epistatic coefficients.
        """
        if self.log_transform:
            self.regression_model = LinearRegression(fit_intercept=False)
            self.regression_model.fit(self.X, self.binary.log.phenotypes)
            self._score = self.regression_model.score(self.X, self.binary.log.phenotypes)
            self.epistasis.values = self.base**(self.regression_model.coef_)
        else:
            self.regression_model = LinearRegression(fit_intercept=False)
            self.regression_model.fit(self.X, self.binary.phenotypes)
            self._score = self.regression_model.score(self.X, self.binary.phenotypes)
            self.epistasis.values = self.regression_model.coef_

    def bootstrap(self, nsamples):
        """Estimate the error in the epistatic coefficients by bootstrapping.
        Draws random samples of the phenotypes from the experimental standard
        error. The main assumption of this method is that the error is normally
        distributed and independent. Sampling is finished once the standard
        deviation.

        Parameters
        ----------
        sample_size : int (default 10)
            number a times to run regression before checking for convergence and
            restarting the another sample.
        rtol : float (default 1e-2)
            tolerance threshold for convergence.
        """
        raise Exception("""Currently broken!""")
        interactions, mean, std, count = resample_to_convergence(self.fit,
            sample_size=sample_size,
            rtol=rtol
        )
        self.epistasis.values = mean
        self.epistasis.stdeviations
