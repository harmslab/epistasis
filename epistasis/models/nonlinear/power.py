import scipy
import numpy as np
import inspect
import json
from .regression import EpistasisNonlinearRegression
from epistasis.stats import gmean

def power_transform(x, lmbda, A, B):
    """Power transformation function. Ignore zeros in gmean calculation"""
    # Check for zeros
    gm = gmean(x+A)
    if lmbda == 0:
        return gm*np.log(x + A)
    else:
        first = (x+A)**lmbda
        out = (first - 1.0)/(lmbda * gm**(lmbda-1)) + B
    return out

class EpistasisPowerTransform(EpistasisNonlinearRegression):
    """"""
    def __init__(self, order=1, model_type="global", fix_linear=False, **kwargs):
        super(EpistasisPowerTransform, self).__init__(
            function=power_transform,
            reverse=self.reverse,
            order=order,
            model_type=model_type,
            fix_linear=fix_linear,
            **kwargs)

        def function(x, lmbda, A, B):
            """Power transformation function. Ignore zeros in gmean calculation"""
            # Check for zeros
            gm = gmean(x+A)
            if lmbda == 0:
                return gm*np.log(x + A)
            else:
                first = (x+A)**lmbda
                out = (first - 1.0)/(lmbda * gm**(lmbda-1)) + B
            return out

        self.function = function

    @property
    def gmean(self):
        linear = np.dot(self.X, self.coef_)
        return gmean(linear + self.parameters.A)

    def reverse(self, y, lmbda, A, B):
        """reverse transform"""
        gmean = self.gmean
        return (gmean**(lmbda-1)*lmbda*(y - B) + 1)**(1/lmbda) - A

    def _params_to_json(self, filename):
        """Temporary method. will be deprecated soon!!!!!
        """
        params = self.parameters()

        params.update(gmean=self.gmean)
        with open(filename, "w") as f:
            json.dump(params, f)
