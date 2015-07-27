import numpy as np

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
    
def chi_squared(y_obs, y_pred):
    """ Calculate the chi squared between observed and predicted y. """
    return sum( (y_obs - y_pred)**2/ y_pred )
    
def log_likelihood_ratio(model1, model2):
    """ Calculate the likelihood ratio two regressed epistasis models. 
    
        Models must be instances of 
    """
    if isinstance(model1) != True and isinstance(model2) != True:
        raise Exception("Models must be instances of the ProjectedEpistasisModel.")
    
    # Calculate the chi-squares of each model.
    chi1 = chi_squared(model1.phenotypes, model1.predict())
    chi2 = chi_squared(model2.phenotypes, model2.predict())
    
    ratio = -2 * np.log(chi1/chi2)
    return ratio