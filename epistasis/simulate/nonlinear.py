import numpy as np
from .multiplicative import MultiplicativeSimulation
from seqspace.gpm import GenotypePhenotypeMap
from .base import BaseSimulation
from epistasis.models.nonlinear import Parameters

class NonlinearSimulation(GenotypePhenotypeMap, BaseSimulation):
    """ Nonlinear function simulation
    """
    def __init__(self, wildtype, mutations, order,
            function,
            coeff_range,
            p0,
            model_type='local',
        ):
        # Linear epistasis map
        self.linear = MultiplicativeSimulation(wildtype, mutations, order, model_type)
        self.function = function

        # Construct GPM
        super(NonlinearSimulation, self).__init__(
            self.linear.wildtype,
            self.linear.genotypes,
            self.linear.phenotypes,
            mutations=self.linear.mutations,
            log_transform=False
        )
        # Get the parameters from the nonlinear function argument list
        function_sign = inspect.signature(self.function)
        parameters = list(function_sign.parameters.keys())


        # Check that the first argument is epistasis
        if parameters[0] != "x":
            raise Exception("""First argument of the nonlinear function must be `x`.""")

        #Set parameters
        self.parameters = Parameters(parameters[1:])
        for i in range(1, len(parameters)):
            self.parameters._set_param(parameters[i], p0[i-1])

        # Build phenotypes.
        self.build()

    def build(self, *args):
        """ Build a set of nonlinear
        """
        self.phenotypes = self.function(self.linear.phenotypes, *self.parameters.values)

    def widget(self, **kwargs):
        """ A widget for playing with different values of nonlinearity. """
        pass
