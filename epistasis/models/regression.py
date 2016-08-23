# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import numpy as np
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------
# seqspace imports
# ------------------------------------------------------------

from seqspace.utils import (list_binary,
                            enumerate_space,
                            encode_mutations,
                            construct_genotypes)


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

class RegressionStats(object):
    """Object for managing all statistics from epistasis regression
    """
    def __init__(self, model):
        self._model = model

    @property
    def score(self):
        """ Get the epistasis model score after estimating interactions. """
        return self._model._score

    def predict(self):
        """ Infer the phenotypes from model.

            __Returns__:

            `genotypes` [array] : array of genotypes -- in same order as phenotypes

            `phenotypes` [array] : array of quantitative phenotypes.
        """
        binaries = self._model.binary.complete_genotypes
        X = generate_dv_matrix(binaries, self._model.epistasis.labels, encoding=self._model.encoding)
        phenotypes = self._model.regression_model.predict(X)
        if self._model.log_transform:
            phenotypes = self._model.base**phenotypes
        return phenotypes


class LinearEpistasisRegression(BaseModel):
    """ Create a map from local epistasis model projected into lower order
    order epistasis interactions. Requires regression to estimate values.

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
            raise Exception("""Need to specify the model's `order` argument or manually
                                list model parameters as `parameters` argument.""")
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

        # Set encoding from model_type given
        self.encoding = model_types[model_type]["encoding"]

        # Construct decomposition matrix
        self.X = generate_dv_matrix(self.binary.genotypes, self.epistasis.labels, encoding=self.encoding)

        # Initialize useful objects to model object
        self.statistics = RegressionStats(self)

        # Try to add plotting module if matplotlib is installed
        try:
            self.plot = RegressionPlotting(self)
        except Warning:
            pass

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

    def fit_error(self, sample_size=10, rtol=1e-2):
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
