import inspect
import numpy as np
from .base import BaseSimulation
from epistasis.models.nonlinear import Parameters


class NonlinearSimulation(BaseSimulation):
    """Constructs a GenotypePhenotypeMap object that exhibits specific
    epistatic interactions and a nonlinear phenotype scale.

    Parameters
    ----------
    wildtype:
    mutations:
    function:
    p0:
    model_type:
    """

    def __init__(self, wildtype, mutations,
                 function=None,
                 p0=[],
                 model_type='global',
                 **kwargs):

        # Set parameters
        self.model_type = model_type
        self.set_function(function, p0=p0)

        # Initialize base class.
        super(NonlinearSimulation, self).__init__(wildtype, mutations,
                                                  **kwargs)

    def set_function(self, function, p0=[]):
        """Set the function that determines the nonlinear phenotypic scale."""
        if function is None:
            raise Exception("Must set `function` kwarg.")
        # Set the nonlinear function
        self.function = function

        # Get the parameters from the nonlinear function argument list
        function_sign = inspect.signature(self.function)
        parameters = list(function_sign.parameters.keys())

        # Check that the first argument is epistasis
        if parameters[0] != "x":
            raise Exception(
                """First argument of the nonlinear function must be `x`.""")

        # Set parameters
        self.parameters = Parameters(parameters[1:])

        for i in range(1, len(parameters)):
            self.parameters._set_param(parameters[i], p0[i - 1])
        return self

    @classmethod
    def from_linear(cls, model, function, p0=[], **kwargs):
        """Layer nonlinear model on top of existing linear model."""
        # Initialize the class
        self = cls(model.wildtype, model.mutations, function, p0=p0, **kwargs)

        # Copy the epistasis map
        self.epistasis = model.epistasis

        # Build phenotypes with a nonlinear scale.
        self.build()
        return self

    def build(self, *args):
        """ Build nonlinear map from epistasis and function.
        """
        # Construct an X for the linear epistasis model
        X = self.add_X()

        # Build linear phenotypes
        self.linear_phenotypes = np.dot(X, self.epistasis.values)

        # Build nonlinear phenotypes
        self.data['phenotypes'] = self.function(_phenotypes, *self.parameters.values)
