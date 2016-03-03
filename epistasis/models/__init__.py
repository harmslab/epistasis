__all__ =  ["linear","nonlinear","pca", "regression", "specifier"]

# Import principal component analysiss
from epistasis.models.pca import EpistasisPCA

# Import linear models
from epistasis.models.linear import (LinearEpistasisModel,
                                    LocalEpistasisModel,
                                    GlobalEpistasisModel)

# Import regression
from epistasis.models.regression import EpistasisRegression

# import nonlinear model
from epistasis.models.nonlinear import NonlinearEpistasisModel

# import epistasis specifier
from epistasis.models.specifier import (LinearEpistasisSpecifier, 
                                        NonlinearEpistasisSpecifier)
                                        
                                        
EPISTASIS_MODELS = [EpistasisPCA, 
    LinearEpistasisModel, 
    LocalEpistasisModel,
    GlobalEpistasisModel,
    EpistasisRegression,
    NonlinearEpistasisModel,
    LinearEpistasisSpecifier,
    NonlinearEpistasisSpecifier 
]