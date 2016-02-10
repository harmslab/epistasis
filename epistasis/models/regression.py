# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import numpy as np
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------
# seqspace imports
# ------------------------------------------------------------

from seqspace.utils import list_binary, enumerate_space, encode_mutations, construct_genotypes

# ------------------------------------------------------------
# Local imports
# ------------------------------------------------------------

from epistasis.decomposition import generate_dv_matrix
from epistasis.models.base import BaseModel
from epistasis.utils import (epistatic_order_indices,
                                build_model_params)

# ------------------------------------------------------------
# Unique Epistasis Functions
# ------------------------------------------------------------

class EpistasisRegression(BaseModel):

    def __init__(self, wildtype, genotypes, phenotypes, 
                order=None, 
                parameters=None, 
                stdeviations=None, 
                log_transform=False, 
                mutations=None, 
                n_replicates=1, 
                model_type="local"):
                
        """ Create a map from local epistasis model projected into lower order
            order epistasis interactions. Requires regression to estimate values.

            __Arguments__:

            `wildtype` [str] : Wildtype genotype. Wildtype phenotype will be used as reference state.

            `genotypes` [array-like, dtype=str] : Genotypes in map. Can be binary strings, or not.

            `phenotypes` [array-like] : Quantitative phenotype values

            `order` [int] : Order of regression; if None, parameters must be passed in manually as parameters=<list of lists>

            `parameters` [dict] : interaction keys with their values expressed as lists.

            `errors` [array-like] : List of phenotype errors.

            `log_transform` [bool] : If True, log transform the phenotypes.

            `mutations` [dict] : site-by-site mutation scheme.

            `model` [str, default='local'] : either 'global' or 'local'. If 'local', wildtype is reference state. If 'global',
                                is cleverly chosen average state.
        """
        # Populate Epistasis Map
        super(EpistasisRegression, self).__init__(wildtype, genotypes, phenotypes, 
                stdeviations=stdeviations, 
                log_transform=log_transform, 
                mutations=mutations, 
                n_replicates=n_replicates)

        # Generate basis matrix for mutant cycle approach to epistasis.
        if order is not None:
            self.order = order
            # Construct the Interactions mapping -- Interactions Subclass is added to model
            self._construct_interactions()
        elif parameters is not None:
            self._construct_interactions()
            self.Interactions.labels = list(parameters.values())
        else:
            raise Exception("""Need to specify the model's `order` argument or manually
                                list model parameters as `parameters` argument.""")

        # Define the encoding for different models
        model_types = {
            "local": {
                "encoding": {"1": 1, "0": 0},       # Decomposition matrix encoding
            }, 
            "global": {
                "encoding": {"1": -1, "0": 1},
            }
        }
        
        # Set encoding from model_type given
        self.encoding = model_types[model_type]["encoding"]

        # Construct decomposition matrix
        self.X = generate_dv_matrix(self.Binary.genotypes, self.Interactions.labels, encoding=self.encoding)


    @property
    def score(self):
        """ Get the epistasis model score after estimating interactions. """
        return self._score


    def fit(self):
        """ Estimate the values of all epistatic interactions using the expanded
            mutant cycle method to any order<=number of mutations.
        """
        self.regression_model = LinearRegression(fit_intercept=False)
        self.regression_model.fit(self.X, self.Binary.phenotypes)
        self._score = self.regression_model.score(self.X, self.Binary.phenotypes)
        self.Interactions.values = self.regression_model.coef_


    def fit_error(self):
        """ Estimate the error of each epistatic interaction by standard error
            propagation of the phenotypes through the model.

            CANNOT propagate error in regressed model.
        """
        self.error_model = LinearRegression(fit_intercept=False)
        pass

    def predict(self):
        """ Infer the phenotypes from model.

            __Returns__:

            `genotypes` [array] : array of genotypes -- in same order as phenotypes

            `phenotypes` [array] : array of quantitative phenotypes.
        """
        phenotypes = np.zeros(len(self.complete_genotypes), dtype=float)
        binaries = self.Binary.complete_genotypes
        X = generate_dv_matrix(binaries, self.Interactions.labels, encoding=self.encoding)
        phenotypes = self.regression_model.predict(X)
        
        return phenotypes
