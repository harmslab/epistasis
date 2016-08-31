import scipy
import numpy

def power_transform(x, lmbda, A, B):
    """Power transformation function."""
    gmean = scipy.stats.mstats.gmean(x + A)
    if lmbda == 0:
        return gmean*numpy.log(x+A)
    else:
        first = (x+A)**lmbda
        out = (first - 1.0)/(lmbda * gmean**(lmbda-1)) + B
        return out
