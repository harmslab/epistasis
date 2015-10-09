import numpy as np

from epistasis.simulate.base import BaseArtificialMap
from epistasis.utils import genotype_params, label_to_key

class ThresholdEpistasisMap(BaseArtificialMap):

    """Generate genotype-phenotype map with thresholding behavior."""

    def __init__(self, length, order, threshold, sharpness, magnitude, allow_neg=True):
        """ Build an epistatic genotype phenotype map with thresholding behavior. Built from
            the function:

                 f(epistasis_model) = \theta - exp(-\nu * epistasis_model)

            where epistasis model is a MULTIPLICATIVE MODEL.

            __Arguments__:

            `length` [int] : length of strings

            `order` [int] : order of epistasis in space

            `threshold` [float] : fitness/phenotype thresholding value.

            `sharpness` [float] : rate of exponential growth towards thresholding value.

            `magnitude` [float] : the limits of the random epistatic terms centered around 1.

            `allow_neg` [bool, default=True] : If false, no deleterious interactions will be set (i.e. < 1.0)

        """
        super(ThresholdEpistasisMap,self).__init__(length, order, log_transform=False)

        high = 1+magnitude
        low = 1-magnitude
        if allow_neg is False:
            low = 1

        vals = self.random_epistasis(low, high, allow_neg=True)
        #vals[0] = 1.0
        self.Interactions.values = vals
        self.raw_phenotypes = self.build_phenotypes()
        self.phenotypes = self.threshold_func(self.raw_phenotypes, threshold, sharpness)

    def build_phenotypes(self, values=None):
        """ Uses the multiplicative model to construct raw phenotypes. """
        phenotypes = np.empty(len(self.genotypes), dtype=float)
        param_map = self.get_map("Interactions.keys", "Interactions.values")
        for i in range(len(phenotypes)):
            params = genotype_params(self.genotypes[i], order=self.order)
            values = np.array([param_map[label_to_key(p)] for p in params])
            phenotypes[i] = np.prod(values)
        return phenotypes

    def threshold_func(self, raw_phenotypes, threshold, sharpness):
        """ Apply the thresholding effect. """
        phenotypes = np.empty(self.n, dtype=float)
        for i in range(self.n):
            phenotypes[i] = threshold * (1 - np.exp(-sharpness*raw_phenotypes[i]))
        return phenotypes
