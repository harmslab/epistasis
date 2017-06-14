import numpy as np
import emcee as emcee
from .base import Sampler

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
        """
        ydata = model.gpm.phenotypes
        yerr = model.gpm.std.upper
        ymodel = model.hypothesis(thetas=coefs)
        inv_sigma2 = 1.0/(yerr**2)
        return -0.5*(np.sum((ydata-ymodel)**2*inv_sigma2 - np.log(inv_sigma2)))

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
        x = lp + BayesianSampler.lnlike(coefs, model)
        # Constrol against Nans -- check if this is too much of a hack later.
        if np.isnan(x).any():
            return -np.inf
        return lp + BayesianSampler.lnlike(coefs, model)

    def add_samples(self, n_mcsteps, nwalkers=None, starting_widths=1e-3):
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

    def predict_from_weighted_samples(self, n):
        """Draw from predicted phenotypes, sampling

        Parameters
        ----------
        n : int
            Number of top models to draw to create a set of predictions.

        Returns
        -------
        predictions : 2d array
            Sets of data predicted from the sampled models.
        """
        sample_size, coef_size = self.coefs.shape
        scores = np.exp(self.scores.value)
        weights = scores / scores.sum()
        model_indices = np.random.choice(np.arange(sample_size), n, replace=True, p=weights)
        samples = np.empty((n, coef_size))
        for i, index in enumerate(model_indices):
            samples[i,:] = self.coefs[index, :]
        return self.predict(samples=samples)


class MixedBayesianSampler(BayesianSampler):
    """
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
        """
        ### Data
        ydata = model.gpm.phenotypes
        yerr = model.gpm.std.upper

        ###  Pull out coefficients
        # Logit coefficients
        logit_coefs = coefs[0:model.Classifier.epistasis.n+1]
        logit_threshold = logit_coefs[0]
        # Epistasis model coefficients
        model_coefs = coefs[model.Classifier.epistasis.n+1:]

        ### log-likelihood of logit model
        prob_1 = model.Classifier.hypothesis(log_coefs)
        ybin = binarize(ydata, threshold)[0]
        lnlike_logit = ybin * np.log(prob_1) + (1 - ybin) * np.log(1-prob_1)

        ### log-likelihood of the epistasis model
        ymodel = model.hypothesis(thetas=model_coefs)
        inv_sigma2 = 1.0/(yerr**2)
        return -0.5*(np.sum((ydata-ymodel)**2*inv_sigma2 - np.log(inv_sigma2))) + lnlike_logit
