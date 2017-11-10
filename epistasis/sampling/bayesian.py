import pandas as pd
import numpy as np
import emcee
import warnings
from functools import wraps

class BayesianSampler(object):
    """A sampling class to estimate the uncertainties in an epistasis model's
    coefficients using a Bayesian approach. This object samples from
    the experimental uncertainty in the phenotypes to estimate confidence
    intervals for the coefficients in an epistasis model according to Bayes Theorem:

    .. math::
        P(H|E) = \\frac{ P(E|H) \cdot P(H) }{ P(E) }

    This reads: "the probability of epistasis model :math:`H` given the data
    :math:`E` is equal to the probability of the data given the model times the
    probability of the model."

    Parameters
    ----------
    model :
        Epistasis model to run a bootstrap calculation.
    """
    def __init__(self, model, lnprior=None, last_run=None):
        # Get needed features from ML model.
        self.model = model
        self.lnlikelihood = model.lnlikelihood
        self.ml_thetas = self.model.thetas
        
        # Set the log-prior function
        if lnprior is not None:
            self.lnprior = lnprior
        
        #### Prepare emcee sampler
        # Get dimensions of the sampler (number of walkers, number of coefs to sample)
        self.ndim = len(self.ml_thetas)
        self.nwalkers = 2*self.ndim
        self.last_run = last_run

        # Construct sampler
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob, args=(self.lnlikelihood,))
        
    @staticmethod
    def lnprior(thetas):
        """Prior probability for the given set of model parameters."""
        return 0.0
        
    @staticmethod
    def lnprob(thetas, lnlike):
        """The posterior probability of a given set of model parameters and likelihood function."""
        lp = BayesianSampler.lnprior(thetas)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(thetas=thetas)
        
    def get_initial_walkers(self, relative_widths=1e-2):
        """Place the walkers in Gaussian balls in parameter space around
        the ML values for each coefficient.
        """
        middle_positions = np.array(self.ml_thetas)
        ### Construct walkers for each coefficient
        deviations = np.random.randn(self.nwalkers, self.ndim)
        
        # Scale deviations appropriately for coefficient magnitude.
        scales = (relative_widths * middle_positions)
        
        # Scale deviations
        rel_deviations = np.multiply(deviations, scales)
        
        # Return walker positions
        walker_positions = middle_positions + rel_deviations
        return walker_positions
        
    def sample(self, n_steps=100, n_burn=0):
        """Sample the likelihood of the model by walking n_steps with each walker."""
        # Suppress warnings that occur when sampling the model.
        warnings.simplefilter("ignore", RuntimeWarning)
        
        # Check if a previous run was given
        if self.last_run is None:
            pos0 = self.get_initial_walkers()
            if n_burn != 0:
                self.sampler.run_mcmc(pos0=pos0, 
                                      N=n_burn, 
                                      storechain=False)
            
            
        else:
            pos0 = self.last_run[0]
            lnprob0 = self.last_run[1]
            rstate0 = self.last_run[2]            
        
            
        
        
        
        if n_burn != 0:
            
            self.sampler.run_mcmc()
            
            
        
        
        
        # Prepare sampler initial conditions. If the sampler was run previously,
        # get ending state and use a initial states.
        if self.last_run is None:
            # Get initial positions of walkers
            self.n_burn = n_burn
            pos0 = self.get_initial_walkers()
            rstate0 = None
            lnprob0 = None
            n_steps = n_steps + n_burn
        else:
            pos0 = self.last_run[0]
            lnprob0 = self.last_run[1]
            rstate0 = self.last_run[2]
            n_steps = n_steps
        
        # Run sampler
        self.last_run = self.sampler.run_mcmc(pos0, n_steps, rstate0=rstate0, lnprob0=lnprob0)
        
        # Store the samples
        samples = self.sampler.chain[:, self.n_burn:, :].reshape((-1, self.ndim))
        index = np.arange(len(samples))
        self.samples = pd.DataFrame(samples, index=index)
        
        return self
            
    def predict(self, store=True):
        """Predict phenotypes from sampled models."""
        # Initialize predictions array
        samples = self.samples
        index = self.samples.index.values
        
        predictions = np.empty((len(self.samples), len(self.model.gpm.complete_genotypes)), dtype=float)
                
        # Begin predicting samples
        for i in samples.index:
            # Slice the row from samples
            thetas = samples.iloc[[i]].values.reshape(-1)
            predictions[i,:] = self.model.hypothesis(X='complete', thetas=thetas)
            
        # Return samples
        df = pd.DataFrame(predictions, index=index, columns=self.model.gpm.complete_genotypes)
        if store == True :
            self.predictions = df
        return df
    
    
