import numpy as np
import itertools as it
from scipy.optimize import curve_fit

# ------------------------------------------
# Possible functions for nonlinear epistasis
# ------------------------------------------

def two_state_func(x, *args):
    """ Two state boltzmann weighted nonlinear function."""
    beta = args[-1]
    params = args[:-1]
    length = len(params)
    params1 = np.array(params[0:int(length/2)])
    params2 = np.array(params[int(length/2):])
    X1 = np.exp(-np.dot(x[0:int(length/2),:].T, params1)/beta)
    X2 = np.exp(-np.dot(x[int(length/2):-1,:].T, params2)/beta)
    return -beta*np.log(X1 + X2)

# ------------------------------------------
# Classes for NonLinear modelling
# ------------------------------------------

class NonlinearEpistasisModel(BaseModel):
        
    def __init__(self, wildtype, genotypes, phenotypes, function, x, parameters, errors=None, log_transform=False, mutations=None):
        """ Fit a nonlinear epistasis model to a genotype-phenotype map. The function and parameters must
            specified prior. 
        
            Uses Scipy's curve_fit function.
            
            Args:
            ----
            wildtype: str
                Wildtype genotype. Wildtype phenotype will be used as reference state.
            genotypes: array-like, dtype=str
                Genotypes in map. Can be binary strings, or not.
            phenotypes: array-like
                Quantitative phenotype values
            function: callable
                Function to fit the linear model. Must have the arguments that correspond to the parameters.
            x: 2d array
                Array of dummy variables for epistasis model fitting (use `generate_dv_matrix` in regression_ext)
            parameters: dict
                interaction keys with their values expressed as lists.
            errors: array-like
                List of phenotype errors.
            log_transform: bool
                If True, log transform the phenotypes.
        """
        # Populate Epistasis Map
        super(NonlinearEpistasisModel, self).__init__(wildtype, genotypes, phenotypes, errors, log_transform, mutations=mutations)
        
        # Generate basis matrix for mutant cycle approach to epistasis.
        if parameters is not None:
            self._construct_interactions()      
            self.Interactions.keys = list(parameters.values())
            self.Interactions.labels = list(parameters.values())
        else:
            raise Exception("""Need to specify the model's `order` argument or manually 
                                list model parameters as `parameters` argument.""")
   
        self.X = x
        self.function = function
    
    def fit(self, p0=None):
        """ Fit the nonlinear function """
        
        # Try initial guess 
        if p0 == None:
            p0 = 0.1*np.ones(len(self.Interactions.labels), dtype=float)
        
        values, cov = curve_fit(self.function, 
                                self.X.T, 
                                self.Binary.phenotypes, 
                                p0=p0, 
                                maxfev=1000000)
                                
        self.Interactions.values = values[:]
        # Setting error if covariance was estimated, else pass.
        try:
            self.errors = cov[:]
        except:
            pass