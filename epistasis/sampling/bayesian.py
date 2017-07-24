import numpy as np
import emcee as emcee
from .base import Sampler, file_handler

class BayesianSampler(Sampler):
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
    db_dir : str (default=None)
        Name a the database directory for storing samples.

    Attributes
    ----------
    coefs : array
        samples for the coefs in the epistasis model.
    scores : array
        Log probabilities for each sample.
    best_coefs : array
        most probable model.
    """
    @staticmethod
    def lnlike(coefs, model):
        """Calculate the log likelihood of a model, given the data.

        Parameters
        ----------
        coefs : array
            All coefficients for an epistasis model. Must be sorted appropriately.
        model :
            Any epistasis model in ``epistasis.models``.

        Returns
        -------
        predictions :
        """
        lnlike = model.lnlikelihood(thetas=coefs)
        return lnlike

    @staticmethod
    def lnprior(coefs):
        """Calculate the probabilities for a set model parameters. This method
        returns a flat prior (log-prior of 0). Redefine this method otherwise.

        Parameters
        ----------
        coefs : array
            All coefficients for an epistasis model. Must be sorted appropriately.
        """
        return 0

    @staticmethod
    def lnprob(coefs, model):
        """Calculate the right hand side of Bayes theorem for an epistasis model. (i.e.
        the "log-likelihood of a model" + "log-prior of a model")

        Parameters
        ----------
        coefs : array
            All coefficients for an epistasis model. Must be sorted appropriately.
        model :
            Any epistasis model in ``epistasis.models``.
        """
        lp = BayesianSampler.lnprior(coefs)
        if not np.isfinite(lp):
            return -np.inf
        lnlike = BayesianSampler.lnlike(coefs, model)
        x = lp + lnlike
        # Constrol against Nans -- check if this is too much of a hack later.
        if np.isnan(x).any():
            return -np.inf
        return x

    @file_handler
    def add_samples(self, n_samples, nwalkers=None, equil_steps=100):
        """Add samples to database"""
        # Calculate the maximum likelihood estimate for the epistasis model.
        try:
            ml_coefs = self.model.thetas
        except AttributeError:
            raise Exception("Need to call the `fit` method to acquire a ML fit first.")

        # Prepare walker number for bayesian sampler
        ndims = len(ml_coefs)
        if nwalkers is None:
            nwalkers = 2 * len(ml_coefs)

        # Calculate the number of steps to take
        mcmc_steps = int(n_samples / nwalkers)

        # Initialize a sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndims, self.lnprob, args=(self.model,))

        # Equilibrate if this if the first time sampling.
        if len(self.coefs) == 0:
            # Construct a bunch of walkers gaussians around each ml_coef
            multigauss_err = 1e-3*np.random.randn(nwalkers, ndims)
            p0 = np.array([ml_coefs for i in range(nwalkers)]) + multigauss_err

            # Run for the number of samples
            pos, prob, state = sampler.run_mcmc(p0, equil_steps, storechain=False)
            sampler.reset()
        else:
            # Start from a previous position
            pos = self.coefs[-nwalkers:,:]

        # Sample
        sampler.run_mcmc(pos, mcmc_steps)

        # Write samples to database
        samples = sampler.flatchain
        scores = sampler.flatlnprobability
        self.write_dataset("coefs", samples)
        self.write_dataset("scores", scores)
