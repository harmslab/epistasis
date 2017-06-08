import numpy as np
from .base import Sampler

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
        self.model.fit(y=pseudo_p)
        coefs = self.model.epistasis.values
        ssr = self.model.score()
        return coefs, ssr

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
        scores = self.scores
        weights = scores / scores.sum()
        model_indices = np.random.choice(np.arange(sample_size), n, replace=True, p=weights)
        samples = np.empty((n, coef_size))
        for i, index in enumerate(model_indices):
            samples[i,:] = self.coefs[index, :]
        return self.predict(samples=samples)
