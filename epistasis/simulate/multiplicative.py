

__doc__ = """Submodule with various classes for generating/simulating genotype-phenotype maps."""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import numpy as np

# local imports
from epistasis.decomposition import generate_dv_matrix
from epistasis.simulate.base import BaseSimulation
from epistasis.mapping.epistasis import EpistasisMap


from seqspace.utils import encode_mutations, construct_genotypes
# ------------------------------------------------------------
# ArtificialMap object can be used to quickly generating a toy
# space for testing the EpistasisModels
# ------------------------------------------------------------

class MultiplicativeSimulation(EpistasisMap, BaseSimulation):
    """ Construct an genotype-phenotype from multiplicative building blocks and
    epistatic coefficients.

    Example
    -------
    Phenotype = b0 * b1 * b2 * b3 * b12 * b13 * b13 * b123
    or
    log(phenotype) = log(b0) + log(b1) + log(b2) + log(b3) + log(b12)
        + log(b13) + log(b13) + log(b123)

    Arguments
    ---------
    wildtype : str
        Wildtype genotype
    mutations : dict
        Mapping for each site to its alphabet
    order : int
        Order of epistasis in simulated genotype-phenotype map
    betas : array-like
        values of epistatic coefficients (must be positive for this function
        to work. Log is taken)
    model_type : str
        Use a local or global (i.e. Walsh space) epistasis model to construct
        phenotypes
    """
    def __init__(self, wildtype,
            mutations,
            order,
            betas,
            model_type='local',
        ):
        # Construct the genotypes first
        encoding = encode_mutations(wildtype, mutations)
        genotypes, binary = construct_genotypes(encoding)

        # Construct epistasis mapping objects (empty)
        super(MultiplicativeSimulation,self).__init__(wildtype, list(genotypes), np.ones(len(genotypes)),
            mutations=mutations,
            log_transform=True
        )
        # Set order, which builds interaction mapping.
        self.order = order
        self._construct_interactions()

        # Betas must be array
        betas = np.array(betas)

        # Check that no betas are negative
        if len(betas[betas<0]) != 0 :
            raise Exception ( """`betas` must be non-negative.""" )

        # Add values to epistatic interactions
        self.Interactions.values = np.log10(betas)

        # Set model type for construction
        self.model_type = model_type

        # build the phenotypes from the epistatic interactions
        self.build()

    def build(self):
        """ Build the phenotype map from epistatic interactions.

            For a multiplicative model, this means log-transforming the phenotypes
            first, using linear system of equations to construct the log-phenotypes,
            then back-transforming the phenotypes to non-log space.
        """
        # Get model type:
        if self.model_type == "local":
            encoding = {"1": 1, "0": 0}

            # Build phenotypes from binary representation of space
            self.X = generate_dv_matrix(self.Binary.genotypes, self.Interactions.labels, encoding=encoding)
            log_phenotypes = np.dot(self.X, self.Interactions.values)

        elif self.model_type == "global":
            encoding = {"1": -1, "0": 1}

            # Build phenotypes from binary representation of space
            self.X = generate_dv_matrix(self.Binary.genotypes, self.Interactions.labels, encoding=encoding)
            log_phenotypes = np.dot( self.X, self.Interactions.values)

        else:
            raise Exception("Invalid model type given.")

        # Set the phenotypes in teh genotype-map, making sure to map them correctly
        # to their binary repentation.
        _phenotypes = np.empty(len(log_phenotypes),dtype=float)
        _phenotypes[self.Binary.indices] = 10**log_phenotypes
        self.phenotypes = _phenotypes
