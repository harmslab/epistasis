import numpy as np

from epistasis.models import ProjectedEpistasisModel

def r_squared(y_obs, y_pred):
    """ Calculate the rquared between the observed and predicted y.
        See wikipedia definition of `coefficient of determination`. 
    """
    # Mean fo the y observed
    y_obs_mean = np.mean(y_obs)
    # Total sum of the squares
    ss_total = sum((y_obs - y_obs_mean)**2)
    # Sum of squares of residuals
    ss_residuals = sum((y_obs - y_pred)**2)
    r_squared = 1 - (ss_residuals/ss_total)
    return r_squared 
    
def ss_residuals(y_obs, y_pred):
    """ calculate residuals """    
    return sum((y_obs - y_pred)**2)

def chi_squared(y_obs, y_pred):
    """ Calculate the chi squared between observed and predicted y. """
    return sum( (y_obs - y_pred)**2/ y_pred )
    
# -----------------------------------------------------------------------
# Comparing two models.
# -----------------------------------------------------------------------    

def log_likelihood_ratio(model1, model2):
    """ Calculate the likelihood ratio two regressed epistasis models. 
    
        Models must be instances of ProjectedEpistasisModel
    """
    if isinstance(model1, ProjectedEpistasisModel) != True and isinstance(model2, ProjectedEpistasisModel) != True:
        raise Exception("Models must be instances of the ProjectedEpistasisModel.")
    
    # Calculate the chi-squares of each model.
    chi1 = chi_squared(model1.phenotypes, model1.predict())
    chi2 = chi_squared(model2.phenotypes, model2.predict())
    
    ratio = -2 * np.log(chi1/chi2)
    return ratio
    
def F_test(model1, model2):
    """ Compare two models. """
    # Check that model1 is nested in model2. Not an intelligent test of this, though.
    if len(model1.Interactions.values) >= len(model2.Interactions.values):
        raise Exception("model1 must be nested in model2.")
    
    # number of observations
    n_obs = len(model1.phenotypes)
    
    # Number of parameters in each model
    p1 = len(model1.Interactions.values)
    p2 = len(model2.Interactions.values)
    
    # Sum of square residuals for each model.
    sse1 = ss_residuals(model1.phenotypes, model1.predict())
    sse2 = ss_residuals(model2.phenotypes, model2.predict())
    
    # F-test
    F = ( (sse1 - sse2) / (p2 - p1) ) / (sse2 / (n_obs - p2 - 1))
    
    return F
    
def false_positive(known, predicted, errors, sigmas=2):
    """ Calculate the false positive rate of predicted. Known, predicted 
        and errors must all be the same length.
        
        Parameters:
        ----------
        known: array-like
            Known values for comparing false positives
        predicted: array-like
            Predicted values
        errors: array-like
            Standard error from model
        sigma: int (default=2)
            How many standard errors away (2 == 0.05 false positive rate)
            
        Returns:
        -------
        Rate: float
            False positive rate in data
    """
    
    N = len(known)
    # Check that known, predicted, and errors are the same size.
    if N != len(predicted) or N != len(errors):
        raise Exception("Input arrays must all be the same size")
     
    false_positive   
    for i in range(N):
        # Calculate bounds
        upper = predicted[i] + sigmas*errors[i]
        lower = predicted[i] - sigmas*errors[i]
        
        # Check false positive rate.
        if known[i] > upper and known[i] < lower:
            false_positive.append(i)
    
    # Calculate false positive rate
    N_fp = len(false_positive)
    rate = N_fp/float(N)
    
    return rate
    
    