import numpy as np
import itertools as it

from gpmap.utils import binary_mutations_map

from .base import BaseSimulation
from epistasis.mapping import EpistasisMap


class NkSimulation(BaseSimulation):
    """ Generate genotype-phenotype map from NK fitness models.

    """
    def __init__(self, length, order,
        coeff_range=(-1, 1),
        distribution=None,
        model_type='local',
        neg_coeffs=True
        ):
        wildtype = "0"*length
        mutations = binary_mutations_map(wildtype, "1"*length)
        # Initialize a genotype-phenotype map
        super(NkSimulation, self).__init__(
            wildtype,
            mutations,
            log_transform=False,
        )
        self.epistasis = EpistasisMap(self)
        # Construct the NK epistasis table.
        self.epistasis._order = order
        keys = np.array(["".join(r) for r in it.product('01', repeat=self.epistasis.order)])
        vals = np.empty(len(keys), dtype=float)
        for i,key in enumerate(keys):
            m = key.count('1')
            vals[i] = np.random.uniform(coeff_range[0], coeff_range[1])
        self.epistasis.keys = keys
        self.epistasis.values = vals
        # Build the genotype-phenotype map.
        self.build()

    @classmethod
    def quick_start(cls, length, order, **kwargs):
        """Construct the genotype-phenotype map"""
        return cls(length, order, **kwargs)

    def build(self):
        """Build phenotypes from NK table
        """
        nk_table = self.epistasis.map("keys", "values")
        # Check for even interaction
        neighbor = int(self.epistasis.order/2)
        if self.epistasis.order%2 == 0:
            pre_neighbor = neighbor - 1
        else:
            pre_neighbor = neighbor

        # Use NK table to build phenotypes
        phenotypes = np.zeros(self.n, dtype=float)
        for i in range(len(self.genotypes)):
            f_total = 0
            for j in range(self.length):
                if j-pre_neighbor < 0:
                    pre = self.genotypes[i][-pre_neighbor:]
                    post = self.genotypes[i][j:neighbor+j+1]
                    f = "".join(pre) + "".join(post)
                elif j+neighbor > self.length-1:
                    pre = self.genotypes[i][j-pre_neighbor:j+1]
                    post = self.genotypes[i][0:neighbor]
                    f = "".join(pre) + "".join(post)
                else:
                    f = "".join(self.genotypes[i][j-pre_neighbor:j+neighbor+1])
                f_total += nk_table[f]
            phenotypes[i] = f_total
        self.phenotypes = phenotypes
