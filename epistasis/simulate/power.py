import inspect
import numpy as np
from .base import BaseSimulation
from epistasis.models.nonlinear import Parameters
from epistasis.models.nonlinear.power import power_transform
from epistasis.matrix import get_model_matrix


class PowerScaleSimulation(BaseSimulation):
    """Creates a GenotypePhenotype map that exhibits a nonlinear shape created
    by a power transform function, and linear high-order epistasis that
    deviates from that scale.

    Parameters
    ----------
    wildtype:
    mutations:
    p0: list of floats
        A list of power-scale parameters. the order must be correct -->
        (lmbda, A, B).
    """

    def __init__(self, wildtype, mutations,
                 p0=[],
                 model_type='global',
                 **kwargs):

        # Set parameters
        self.model_type = model_type

        # Set the parameters -- this logic is ugly, I know. Parameters object
        # needs to be refactored and this will be cleaned up. low priority.
        self.parameters = Parameters()
        self.parameters.add(name="lmbda", value=p0[0])
        self.parameters.add(name="A", value=p0[1])
        self.parameters.add(name="B", value=p0[2])

        # Initialize base class.
        super(PowerScaleSimulation, self).__init__(wildtype, mutations,
                                                   **kwargs)

    @staticmethod
    def function(x, lmbda, A, B):
        """Power transform function."""
        return power_transform(x, lmbda, A, B)

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
        self.epistasis.values[0] = self.parameters['B']

        # Construct an X for the linear epistasis model
        X = self.add_X()

        # Build linear phenotypes
        self.linear_phenotypes = np.dot(X, self.epistasis.values)

        # Build nonlinear phenotypes
        self.data['phenotypes'] = self.function(
            self.linear_phenotypes, *self.parameters.values())
