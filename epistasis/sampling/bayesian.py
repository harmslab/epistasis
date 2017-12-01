import pandas as pd
import numpy as np
import emcee
import warnings
from functools import wraps


class BayesianSampler(object):
    """A sampling class to estimate the uncertainties in an epistasis model's
    coefficients using a Bayesian approach. This object samples from
    the experimental uncertainty in the phenotypes to estimate confidence
    intervals for the coefficients in an epistasis model according to Bayes
    Theorem:

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

    def __init__(self, model, lnprior=None):
        # Get needed features from ML model.
        self.model = model
        self.lnlikelihood = model.lnlikelihood
        self.ml_thetas = self.model.thetas

        # Set the log-prior function
        if lnprior is not None:
            self.lnprior = lnprior

        # Prepare emcee sampler
        # Get dimensions of the sampler (number of walkers, number of coefs to
        # sample)
        self.ndim = len(self.ml_thetas)
        self.nwalkers = 2 * self.ndim

        # Construct sampler
        self.sampler_engine = emcee.EnsembleSampler(
            self.nwalkers, self.ndim, self.lnprob, args=(self.lnlikelihood,))

    @staticmethod
    def lnprior(thetas):
        """Prior probability for the given set of model parameters."""
        return 0.0

    @staticmethod
    def lnprob(thetas, lnlike):
        """The posterior probability of a given set of model parameters and
        likelihood function."""
        lp = BayesianSampler.lnprior(thetas)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(thetas=thetas)

    def get_initial_walkers(self, relative_widths=1e-2):
        """Place the walkers in Gaussian balls in parameter space around
        the ML values for each coefficient.
        """
        middle_positions = np.array(self.ml_thetas)
        # Construct walkers for each coefficient
        deviations = np.random.randn(self.nwalkers, self.ndim)

        # Scale deviations appropriately for coefficient magnitude.
        scales = (relative_widths * middle_positions)

        # Scale deviations
        rel_deviations = np.multiply(deviations, scales)

        # Return walker positions
        walker_positions = middle_positions + rel_deviations
        return walker_positions

    def sample(self, n_steps=100, n_burn=0, previous_state=None):
        """Sample the likelihood of the model by walking n_steps with each
        walker."""
        # Suppress warnings that occur when sampling the model.
        warnings.simplefilter("ignore", RuntimeWarning)

        # Check if a previous run was given
        if previous_state is None:
            # Get initialize positions
            pos = self.get_initial_walkers()

            # Run the MCMC walks, burning these states to equilibrate.
            if n_burn != 0:
                pos, lnprob, rstate = self.sampler_engine.run_mcmc(
                    pos0=pos, N=n_burn, storechain=False)
            else:
                lnprob, rstate = None, None
        else:
            # Get previous state.
            pos = previous_state['pos']
            lnprob = previous_state['lnprob']
            rstate = previous_state['rstate']

        # Run sampler from previous position
        pos, lnprob, rstate = self.sampler_engine.run_mcmc(pos0=pos,
                                                           N=n_steps,
                                                           rstate0=rstate,
                                                           lnprob0=lnprob,
                                                           storechain=True)

        # Store previous run in a dictionary
        previous_state = {'pos': pos, 'lnprob': lnprob, 'rstate': rstate}
        return self.sampler_engine.flatchain, previous_state
