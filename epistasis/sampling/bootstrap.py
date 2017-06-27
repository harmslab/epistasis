import numpy as np
from .base import Sampler, file_handler

class BootstrapSampler(Sampler):
    """A sampling class to estimate the uncertainties in an epistasis model's
    coefficients using a bootstrapping method. This object samples from
    the experimental uncertainty in the phenotypes to estimate confidence
    intervals for the coefficients in an epistasis model.

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
        R-squared of each model
    best_coefs : array
        Best fit model.
    """
    def sample(self):
        """"""
        pseudo_p = np.random.normal(loc=self.model.gpm.phenotypes, scale=self.model.gpm.err.upper)
        try:
            parameters = self.model.parameters()
            parameters["B"] = -10
            self.model.fit(X=None, y=pseudo_p, **parameters)
        except:
            self.model.fit(X=None, y=pseudo_p)
        coefs = self.model.epistasis.values
        ssr = self.model.score()
        return coefs, ssr

    @file_handler
    def add_samples(self, n):
        """Add samples to database."""
        # Create a first sample.
        coef, ssr = self.sample()
        samples = np.empty((n, len(coef)),dtype=float)
        samples[0, :] = coef
        scores = np.empty(n,dtype=float)
        scores[0] = ssr
        # Generate more samples.
        for i in range(1, n):
            sample, ssr = self.sample()
            samples[i,:] = sample
            scores[i] = ssr
        # Write samples to database
        self.write_dataset("coefs", samples)
        self.write_dataset("scores", scores)
        return samples
