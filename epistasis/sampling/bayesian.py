"""
A Bayesian, high-order, linear epistasis model.

"""
import numpy as np
import emcee as emcee
from .base import Sampler

class BayesianSampler(Sampler):
    """
    """
    def __init__(self, model, db_dir=None):
        super(BayesianSampler, self).__init__(model, db_dir=None)
        self.File.create_dataset("coefs", (0,0), maxshape=(None,None), compression="gzip")
        self.File.create_dataset("scores", (0,), maxshape=(None,), compression="gzip")

    @staticmethod
    def lnlike(coefs, model):
        """Calculate the log likelihood of a model, given the data."""
        ydata = model.gpm.phenotypes
        yerr = model.gpm.std.upper
        ymodel = model.function(coefs=coefs)
        inv_sigma2 = 1.0/(yerr**2)
        return -0.5*(np.sum((ydata-ymodel)**2*inv_sigma2 - np.log(inv_sigma2)))

    @staticmethod
    def lnprior(coefs):
        """Flat prior"""
        return 0

    @staticmethod
    def lnprob(coefs, model):
        """"""
        lp = BayesianSampler.lnprior(coefs)
        if not np.isfinite(lp):
            return -np.inf
        x = lp + BayesianSampler.lnlike(coefs, model)
        return lp + BayesianSampler.lnlike(coefs, model)

    def add_samples(self, n_mcsteps, nwalkers=None, starting_widths=1e-4):
        """Add samples to database"""
        # Calculate the maximum likelihood estimate for the epistasis model.
        self.model.fit()
        ml_coefs = self.model.epistasis.values

        # Prepare walker number for bayesian sampler
        ndims = len(ml_coefs)
        if nwalkers is None:
            nwalkers = 2 * len(ml_coefs)

        # Construct a bunch of walkers gaussians around each ml_coef
        multigauss_err = starting_widths*np.random.randn(nwalkers, ndims)
        pos = np.array([ml_coefs for i in range(nwalkers)]) + multigauss_err

        # Construct MCMC Sampler using emcee
        sampler = emcee.EnsembleSampler(nwalkers, ndims, self.lnprob, args=(self.model,))

        # Run for the number of samples
        sampler.run_mcmc(pos, n_mcsteps)

        # Write samples to database
        samples = sampler.flatchain
        scores = sampler.flatlnprobability
        self.write_dataset("coefs", samples)
        self.write_dataset("scores", scores)

    @property
    def ml_coefs(self):
        """Most probable model."""
        index = np.argmax(self.scores.value)
        return self.coefs[index,:]

    def percentiles(self, percentiles):
        return np.percentile(self.coefs.value, percentiles, axis=0)
