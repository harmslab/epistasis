import scipy
import numpy as np

from .base import NonlinearEpistasisModel, NonlinearStats

def power_transform(x, lmbda, A, B):
    """Power transformation function."""
    gmean = scipy.stats.mstats.gmean(x + A)
    if lmbda == 0:
        return gmean*np.log(x+A)
    else:
        first = (x+A)**lmbda
        out = (first - 1.0)/(lmbda * gmean**(lmbda-1)) + B
        return out

class PowerTransformStats(NonlinearStats):

    @property
    def gmean(self):
        return scipy.stats.mstats.gmean(self.linear + self.parameters.A)

    def transform(self, x):
        """Power transformation function."""
        gmean = self.gmean
        A = self.parameters.A
        B = self.parameters.B
        lmbda = self.parameters.lmbda
        if lmbda == 0:
            return gmean*np.log(x+A)
        else:
            first = (x+A)**lmbda
            out = (first - 1.0)/(lmbda * gmean**(lmbda-1)) + B
            return out

    def reverse_transform(self, y):
        """reverse transform"""
        gmean = self.gmean
        A = self.parameters.A
        B = self.parameters.B
        lmbda = self.parameters.lmbda
        return (gmean**(lmbda-1)*lmbda*(y - B) + 1)**(1/lmbda) - A


class PowerTransformEpistasisModel(NonlinearEpistasisModel):
    """A model subclass of `NonlinearEpistasisModel` which uses a power-transform
    function to fit the nonlinearity in the genotype-phenotype map.

    Note, power transform function calculates the geometric mean of the un-transformed
    linear epistasis model (also acting as a constraint on the fit). In the base class
    this value floats as a parameter in the fit but is not returned by the fit. This
    makes it difficult to back transform. This model fixes this problem and also
    includes the `transformed` phenotypes as a property in the `statistics` object.

    Parameters
    ----------
    wildtype : str
        wildtype sequence to be used as the reference state.
    genotypes : array-like
        list of genotypes
    phenotypes : array-like
        list of the phenotypes in same order as their genotype
    function : callable
        nonlinear function for scaling phenotypes
    order : int
        order of epistasis model
    stdeviations : array-like
        standard deviations
    log_transform : bool
        if true, log transform the linear space. Note: this does not transform the
        nonlinear feature of this space.
    mutations : dict
        mapping sites to their mutation alphabet
    n_replicates : int
        number of replicate measurements for each phenotypes
    model_type : str
        model type (global or local)
    logbase : callable
        logarithm function for transforming phenotypes.

    Attributes
    ----------
    see seqspace for attributes from GenotypePhenotype.

    parameters : Parameters object
        store output from the nonlinear function parameters
    linear : EpistasisRegression
        linear epistasis regression for calculating specific interactions.
    """
    def __init__(self, wildtype, genotypes, phenotypes,
        order=None,
        stdeviations=None,
        log_transform=False,
        mutations=None,
        n_replicates=1,
        model_type="local",
        logbase=np.log10,
        fix_linear=False):

        super(PowerTransformEpistasisModel, self).__init__(wildtype, genotypes, phenotypes,
            function=power_transform,
            order=order,
            stdeviations=stdeviations,
            log_transform=log_transform,
            mutations=mutations,
            n_replicates=n_replicates,
            model_type=model_type,
            logbase=logbase,
            fix_linear=fix_linear
        )
        self.statistics = PowerTransformStats(self)
