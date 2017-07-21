import inspect
import numpy as np
from .base import BaseSimulation
from epistasis.models.nonlinear import Parameters
from epistasis.model_matrix_ext import get_model_matrix

class NonlinearSimulation(BaseSimulation):
    """ Nonlinear epistasis simulator. Creates a Genotype-Phen"""
    def __init__(self, wildtype, mutations, function,
        p0=[],
        model_type='global',
        **kwargs):
        # Initialize base class.
        super(NonlinearSimulation, self).__init__(wildtype, mutations,
            **kwargs
        )
        self.model_type = model_type
        self.set_function(function, p0=p0)

    def set_function(self, function, p0=[]):
        """Set the nonlinear function.
        """
        # Set the nonlinear function
        self.function = function
        # Get the parameters from the nonlinear function argument list
        function_sign = inspect.signature(self.function)
        parameters = list(function_sign.parameters.keys())
        # Check that the first argument is epistasis
        if parameters[0] != "x":
            raise Exception("""First argument of the nonlinear function must be `x`.""")
        # Set parameters
        self.parameters = Parameters(parameters[1:])
        for i in range(1, len(parameters)):
            self.parameters._set_param(parameters[i], p0[i-1])

    @property
    def p_additive(self):
        """Get the linear, additive phenotypes"""
        orders = self.epistasis.get_orders(0,1)
        x = get_model_matrix(self.binary.genotypes, orders.sites, model_type=self.model_type)
        linear = np.dot(x, orders.values)
        return linear

    @classmethod
    def from_linear(cls, model, function, p0=[], **kwargs):
        """Layer nonlinear model on top of existing linear model."""
        self = cls(model.wildtype, model.mutations, function, p0=p0, **kwargs)
        self.epistasis = model.epistasis
        self.build()
        return self

    def build(self, *args):
        """ Build nonlinear map from epistasis and function.
        """
        # Get model type:
        self.X = get_model_matrix(self.binary.genotypes, self.epistasis.sites, model_type=self.model_type)
        _phenotypes = np.dot(self.X, self.epistasis.values)
        self.phenotypes = self.function(_phenotypes, *self.parameters.values)
