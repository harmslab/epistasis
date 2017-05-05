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

    def sample(self):
        """"""
        pseudo_p = np.random.normal(loc=self.model.gpm.phenotypes, scale=self.model.gpm.err.upper)
        self.model.fit(y=pseudo_p)
        coefs = self.model.epistasis.values
        return coefs

    def add_samples(self, n):
        """Add samples to database
        """
        # Create a first sample.
        coef = self.sample()
        samples = np.empty((n, len(coef)),dtype=float)
        samples[0, :] = coef
        # Generate more samples.
        for i in range(1, n):
            samples[i,:] = self.sample()
        # Write samples to database
        self.write_samples("coefs", samples)

    @property
    def coefs(self):
        """Samples of epistatic coefficients. Rows are samples, Columns are coefs."""
        return self.File["coefs"]

    @property
    def labels(self):
        return self.model.epistasis.labels

    def predict_from_samples(self, coefs_samples=None):
        """"""
        if coefs_samples is None:
            coefs_samples = self.coefs.value
