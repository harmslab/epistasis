import inspect
import lmfit
from lmfit import Parameter, Parameters

from abc import ABC, abstractmethod

# ---------------------- Abstract Minimizer class -------------------------

class Minimizer(ABC):
    """Abstract Base Class for function minimizers.

    Wraps a function. Fits data using the function. Then, that fitted function
    can be used to transform data.
    """
    def __init__(self, function, **p0):
        pass

    @abstractmethod
    def function(self, x, **params):
        """Execute function given parameters.
        """

    @abstractmethod
    def predict(self, x):
        """Predict y, given x, from the fitted model.
        """

    @abstractmethod
    def transform(self, x, y):
        """Method that transforms y onto the x scale using the minimized fit.
        """

    @abstractmethod
    def fit(self, x, y):
        """Fit the x, y data using the function. Store fit.
        """

# ---------------------- Abstract Minimizer class -------------------------


class FunctionMinimizer(Minimizer):
    """Minimizer object that fits data using a function.
    """
    def __init__(self, function, **p0):
        # Do some inspection to get the parameters from the nonlinear
        # function argument list.
        func_signature = inspect.signature(function)
        func_params = list(func_signature.parameters.keys())

        if func_params[0] != "x":
            raise Exception("First argument of the nonlinear function must "
                            "be `x`.")

        # Construct lmfit parameters object
        self.parameters = Parameters()
        for p in func_params[1:]:
            # Get starting value of parameter if given.
            val = None
            if p in p0:
                val = p0[p]
            # Add parameter.
            self.parameters.add(name=p, value=val)

        # Set function
        self._function = function

    def function(self, x, *args, **kwargs):
        """Execute the function."""
        return self._function(x, *args, **kwargs)

    def predict(self, x):
        """Call function"""
        return self._function(x, **self.parameters)

    def transform(self, x, y):
        """Transform y onto the x scale."""
        ymodel = self.predict(x)
        return (y - ymodel) + x

    def fit(self, x, y):
        """Fit the function"""
        # Store residual steps in case fit fails.
        last_residual_set = None

        # Residual function to minimize.
        def residual(params, func, x, y=None):
            # Fit model
            parvals = list(params.values())
            ymodel = func(x, *parvals)

            # Store items in case of error.
            nonlocal last_residual_set
            last_residual_set = (params, ymodel)

            return y - ymodel

        # Minimize the above residual function.
        try:
            self.minimizer = lmfit.minimize(
                residual,
                self.parameters,
                args=[self._function, x],
                kws={'y': y})

            # Point to nonlinear.
            self.parameters = self.minimizer.params

        # If fitting fails, print what happened
        except Exception as e:
            # if e is ValueError
            print("ERROR! Some of the transformed phenotypes are invalid.")
            print("\nParameters:")
            print("----------")
            print(last_residual_set[0].pretty_print())
            print("\nTransformed phenotypes:")
            print("----------------------")
            print(last_residual_set[1])
            raise e
