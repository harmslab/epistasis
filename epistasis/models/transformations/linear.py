# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import numpy as np
import scipy as sp
from functools import wraps

# ------------------------------------------------------------
# Local imports
# ------------------------------------------------------------

from epistasis.stats import pearson
from epistasis.decomposition import generate_dv_matrix
from epistasis.models.base import BaseModel as _BaseModel
from epistasis.models.base import X_fitter
from epistasis.models.base import X_predictor

# ------------------------------------------------------------
# Epistasis Mapping Classes
# ------------------------------------------------------------

class EpistasisLinearTransformation(_BaseModel):
    """A classical linear, high-order epistatic transformation. This is the same
    as doing a full-order EpistasisLinearRegression. This may be more numerically
    stable than regression.

    Use this object to decompose a genotype-phenotype map into additive and
    epistatic (to N-th order) coefficients. This assumes the genotype-phenotype
    map is linear.

    .. math::

        Phenotype = K_0 + \sum_{i=1}^{L} K_i + \sum_{i  j}^{L} K_ij + \sum_{i < j < k }^{L} K_ijk + ...

    Example
    -------
    Create an instance of the model object by passing in the genotype and
    phenotype arguments to the class.

    Attributes
    ----------
    See seqspace for the full list of attributes in the GenotypePhenotypeMap objects.

    plot : object
        A subobject with methods to plot epistatic data. (see ``epistasis.plotting.linear``)
    """
    def __init__(self, model_type="global", **kwargs):
        self.model_type = model_type

    @wraps(_BaseModel.attach_gpm)
    def attach_gpm(self, gpm):
        super(EpistasisLinearTransformation, self).attach_gpm(gpm)
        self.order = self.gpm.binary.length

    @X_fitter
    def fit(self, X=None, y=None):
        """Estimate the values of all epistatic interactions using the expanded
        mutant cycle method to order=number_of_mutations.
        """
        self.coef_ = sp.linalg.solve(X,y)

    @X_predictor
    def predict(self, X=None):
        """Predict y.
        """
        return X.dot(self.coef_)

    @X_fitter
    def score(self, X=None, y=None):
        y_pred = X.dot(self.coef_)
        return pearson(y, y_pred)**2
