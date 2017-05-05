import numpy as np
from .base import Sampler

class BootstrapSampler(Sampler):
    """Bootstrap the uncertainties in a model's coefficients. Sample from
    the experimental uncertainty in the phenotypes to determine confidence
    intervals of the coefficients of an epistasis model.
    """
    def __init__(self, model, db_dir=None):
        super(BootstrapSampler, self).__init__(model, db_dir=None)
        self.File.create_dataset("coefs", (0,0), maxshape=(None,None), compression="gzip")
        self.File.create_dataset("scores", (0,), maxshape=(None,), compression="gzip")

    def sample(self):
        """"""
        pseudo_p = np.random.normal(loc=self.model.gpm.phenotypes, scale=self.model.gpm.err.upper)
        self.model.fit(y=pseudo_p)
        coefs = self.model.epistasis.values
        ssr = self.model.score()
        return coefs, ssr

    def add_samples(self, n):
        """Add samples to database
        """
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
