import numpy as np
import itertools as it

from epistasis.simulate.base import BaseArtificialMap

class NKEpistasisMap(BaseArtificialMap):

    """ Generate genotype-phenotype map from NK fitness models. """

    def __init__(self, length, order, magnitude):
        """ Construct a genotype phenotype map from NK epistatic landscape. """
        super(NKEpistasisMap,self).__init__(length, order, log_transform=False)

        # Construct a NK table
        self.nk_table = self.build_nk_table(magnitude)

        # Use binary genotypes to set the phenotypes using NK table
        self.Binary.phenotypes = self.build()

        # Reorder the phenotypes properly
        phenotypes = np.zeros(len(self.Binary.phenotypes), dtype=float)
        for i in range(len(self.Binary.indices)):
            phenotypes[self.Binary.indices[i]] = self.Binary.phenotypes[i]
        self.phenotypes = phenotypes

    def build_nk_table(self, magnitude):
        """ Returns an nk fitness distribution """

        # Build an NK fitness table
        nk_table = dict()
        interactions = ["".join(r) for r in it.product('01', repeat=self.order)]
        for s in interactions:
            m = s.count('1')
            nk_table[s] = m*magnitude*np.random.rand()*(-1)**np.random.randint(10)
        return nk_table

    def build(self):
        """ Build phenotypes from NK table"""

        # Check that nk table exists
        if hasattr(self, "nk_table") == False:
            raise Exception("NK table must be constructed before building phenotypes.")

        # Check for even interaction
        neighbor = int(self.order/2)
        if self.order%2 == 0:
            pre_neighbor = neighbor - 1
        else:
            pre_neighbor = neighbor

        # Use NK table to build phenotypes
        phenotypes = np.zeros(self.n, dtype=float)
        for i in range(len(self.Binary.genotypes)):
            f_total = 0
            for j in range(self.length):
                if j-pre_neighbor < 0:
                    pre = self.Binary.genotypes[i][-pre_neighbor:]
                    post = self.Binary.genotypes[i][j:neighbor+j+1]
                    f = "".join(pre) + "".join(post)
                elif j+neighbor > self.length-1:
                    pre = self.Binary.genotypes[i][j-pre_neighbor:j+1]
                    post = self.Binary.genotypes[i][0:neighbor]
                    f = "".join(pre) + "".join(post)
                else:
                    f = "".join(self.Binary.genotypes[i][j-pre_neighbor:j+neighbor+1])
                f_total += self.nk_table[f]
            phenotypes[i] = f_total

        return phenotypes
