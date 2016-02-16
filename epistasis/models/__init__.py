__all__ =  ["linear","nonlinear","pca"]

# Import principal component analysiss
from epistasis.models.pca import EpistasisPCA

# Import linear models
from epistasis.models.linear import (LocalEpistasisModel,
                                    GlobalEpistasisModel)

# Need to remove this later
from epistasis.models.regression import EpistasisRegression as ProjectedEpistasisModel
