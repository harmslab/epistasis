from .base import EpistasisLinearTransformation

class LocalEpistasisModel(EpistasisLinearTransformation):
    """Construct a genotype-phenotype map and fit with a linear epistasis model,
    defined as follows.

    P = K_0 + sum(K_i) + sum(K_ij) + sum(K_ijk) + ...

    Parameters
    ----------
    wildtype : str
        Wildtype genotype. Wildtype phenotype will be used as reference state.
    genotypes : array-like, dtype=str
        Genotypes in map. Can be binary strings, or not.
    phenotypes : array-like
        Quantitative phenotype values
    stdeviations : array-like
        List of phenotype errors.
    log_transform : bool
        If True, log transform the phenotypes.
    mutations : dict
        Mapping sites to mutational alphabet.
    n_replicates : int
        number of replicates.
    model_type : str
        type of model to use.
    logbase : callable
        log spaces to transform phenotypes.
    """
    def __init__(self, wildtype, genotypes, phenotypes,
                stdeviations=None,
                log_transform=False,
                mutations=None,
                n_replicates=1,
                logbase=np.log10):
        # Populate Epistasis Map
        super(LocalEpistasisModel, self).__init__(wildtype, genotypes, phenotypes,
                stdeviations=stdeviations,
                log_transform=log_transform,
                mutations=mutations,
                n_replicates=n_replicates,
                model_type="local",
                logbase=logbase)
