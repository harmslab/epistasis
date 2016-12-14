import scipy
import numpy as np
import inspect
from .regression import EpistasisNonlinearRegression

def power_transform(x, lmbda, A, B):
    """Power transformation function."""
    gmean = scipy.stats.mstats.gmean(x+A)
    if lmbda == 0:
        return gmean*np.log(x+A)
    else:
        first = (x+A)**lmbda
        out = (first - 1.0)/(lmbda * gmean**(lmbda-1)) + B
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
            """Power transformation function."""
            gmean = self.gmean
            if lmbda == 0:
                return gmean*np.log(x+A)
            else:
                first = (x+A)**lmbda
                out = (first - 1.0)/(lmbda * gmean**(lmbda-1)) + B
                return out
        self.function = function

    @property
    def gmean(self):
        linear = np.dot(self.X, self.coef_)
        return scipy.stats.mstats.gmean(linear + self.parameters.A)

    def reverse(self, y, lmbda, A, B):
        """reverse transform"""
        gmean = self.gmean
        return (gmean**(lmbda-1)*lmbda*(y - B) + 1)**(1/lmbda) - A
