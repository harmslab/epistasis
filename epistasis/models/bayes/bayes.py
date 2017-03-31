"""
A Bayesian, high-order, linear epistasis model.

"""
import numpy as _np
import emcee as _emcee

from ..base import BaseModel as _BaseModel
from ..linear.regression import EpistasisLinearRegression as _EpistasisLinearRegression
from ..base import X_fitter, X_predictor

class EpistasisLinearBayesianEstimator(_EpistasisLinearRegression, _BaseModel):
    """
    """
    def __init__(self, order=1, model_type="global", **kwargs):
        self.order = order

    @X_fitter
    def log_likelihood(self, betas=None, X=None, y=None, yerr=None):
        """Compute the log likelihood of a given set of parameters and data.
        """
        # Calculate model output
        beta_vec = _np.array(betas)
        ymodel = _np.dot(X, beta_vec)
        # Calculate the sigma
        inv_sigma2 = 1.0/(yerr**2 + model**2*_np.exp(2*lnf))
        return -0.5*(_np.sum((y-ymodel)**2*inv_sigma2 - _np.log(inv_sigma2)))

    def log_prior(self, wildtype, betas):
        """Compute the log prior value for the given set of coefficients.
        """
        return 1

    @X_fitter
    def log_prob(self, betas=None, X=None, y=None, yerr=None):
        """Compute the log probability of a given model using the log prior
        and log likelihood functions.
        """
        return self.log_prior() + self.log_likelihood(betas, X, y, yerr)

    @X_fitter
    def fit(self, X=None, y=None, n_samples=100, n_walkers=None, **kwargs):
        """Use Monte Carlo Markov Chain
        """
        # Compute dimensions for MCMC estimator
        n_coefs =  X.shape[1]
        if n_walkers == None:
            n_walkers = X.shape[1]*2

        # Use the maximum likelihood estimator to start with a set of guesses.
        super(EpistasisLinearBayes, self).fit(X=X,y=None)
        ml_guess = self.epistasis.values
        starting_position = [self.epistasis.values + _np.random.randn(n_coefs) for i in range(n_walkers)]

        # Initialize an emcee Monte Carlo Markov Chain object
        args = (X, y, yerr)
        self.sampler = _emcee.EnsembleSampler(n_walkers, n_coefs, self.log_prob, args=args)
        self.sampler.run_mcmc(starting_position, n_samples, **kwargs)



    @X_predictor
    def predict(self, X=None):
        """
        """
        pass
