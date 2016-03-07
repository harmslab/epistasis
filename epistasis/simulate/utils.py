import numpy as np

def remove_epistasis(model, order):
    """ Truncate epistasis from a linear epistasis model
         and construct a genotype-phenotype from the 
         truncated model.
    
        Arguments:
        ---------
        model: LinearEpistasisModel
            
        order: int, 
            order of epistasis at which the model is to be truncated.    
    """
    # Find last index of the order of epistasis
    index = len([order for key in model.Interactions.labels if len(key) <= order])
    
    # Get truncated X
    X = model.X[:,:index]
    coeffs = model.Interactions.values[:index]
    
    # Compute new phenotypes
    phenotypes = np.dot(X, coeffs)
    
    # handle log transform
    if model.log_transform:
        phenotypes = 10**phenotypes
    
    # Reconstruct genotype-phenotype map
    wildtype = model.wildtype
    genotypes = model.genotypes
    log_transform = model.log_transform
    stdeviations = model.stdeviations
    mutations = model.mutations
    n_replicates = model.n_replicates
    
    return GenotypePhenotypeMap(wildtype, genotypes, phenotypes,
        stdeviations=stdeviations,
        log_transform=log_transform,
        mutations=mutations,
        n_replicates=n_replicates)