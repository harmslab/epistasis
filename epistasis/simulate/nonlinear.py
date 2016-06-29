import numpy as np
from .multiplicative import MultiplicativeSimulation
from seqspace.gpm import GenotypePhenotypeMap


class NonlinearEpistasisMap(GenotypePhenotypeMap):
    """ Nonlinear function simulation
    """
    def __init__(self, wildtype,
            mutations,
            order,
            betas,
            function,
            p0,
            model_type='local',
        ):
        # Linear epistasis map
        self.Linear = MultiplicativeSimulation(wildtype, mutations, order, betas, model_type)
        self.function = function

        # Construct GPM
        super(NonlinearEpistasisMap, self).__init__(
            self.Linear.wildtype,
            self.Linear.genotypes,
            np.ones(len(self.Linear.genotypes), dtype=float),
            mutations=self.Linear.mutations,
            log_transform=False
        )
        # Build phenotypes.
        self.build(*p0)

    def build(self, *args):
        """ Build a set of nonlinear
        """
        if self.Linear.log_transform:
            self.phenotypes = self.function(self.Linear.Raw.phenotypes, *args)
        else:
            self.phenotypes = self.function(self.Linear.phenotypes, *args)


    def widget(self, **kwargs):
        """ A widget for playing with different values of nonlinearity. """
        pass
