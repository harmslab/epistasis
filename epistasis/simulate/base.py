import numpy as np
from gpmap.gpm import GenotypePhenotypeMap
from gpmap import utils
from epistasis.mapping import EpistasisMap, assert_epistasis

class BaseSimulation(GenotypePhenotypeMap):
    """ Base class for simulating genotype-phenotype maps built from epistatic
    interactions.

    Parameters
    ----------
    wildtype : str
        wildtype sequence.
    mutations : dict
        dictionary mapping each site the possible mutations
    """
    def __init__(self, wildtype, mutations,
        log_transform=False,
        logbase=np.log10,
        model_type="global",
        **kwargs
        ):
        genotypes = np.array(utils.mutations_to_genotypes(wildtype, mutations))
        phenotypes = np.ones(len(genotypes))
        # Initialize a genotype-phenotype map
        super(BaseSimulation, self).__init__(
            wildtype,
            genotypes,
            phenotypes,
            log_transform=log_transform,
            logbase=logbase,
            mutations=mutations,
            **kwargs
        )

    def set_coefs_order(self, order):
        """Set coefs from an epistatic order."""
        # Attach an epistasis model.
        self.epistasis = EpistasisMap.from_mutations(self.mutations, order)

    def set_coefs_sites(self, sites):
        """Set coefs from list of coefs sites.
        """
        order = max([len(s) for s in sites])
        self.epistasis = EpistasisMap(sites, order=order, model_type="global")

    @assert_epistasis
    def set_coefs(self, sites, values):
        """Set the epistatic coefs

        Parameters
        ----------
        sites : List
            List of epistatic coefficient sites.
        values : List
            list of floats representing to epistatic coefficients.
        """
        self.set_coefs_sites(sites)
        self.epistasis.values = values
        self.build()

    @assert_epistasis
    def set_coefs_values(self, values):
        """Set coefficient values.
        """
        self.epistasis.values = values
        self.build()

    @assert_epistasis
    def set_coefs_random(self, coef_range):
        """Set coefs to values drawn from a random, uniform distribution between
        coef_range.

        Parameters
        ----------
        coef_range : 2-tuple
            low and high bounds for coeff values.
        """
        # Add values to epistatic interactions
        self.epistasis.values = np.random.uniform(coef_range[0], coef_range[1], size=len(self.epistasis.sites))
        self.build()

    @classmethod
    def from_length(cls, length, **kwargs):
        """Constructs genotype from binary sequences with given length and
        phenotypes from epistasis with a given order.

        Parameters
        ----------
        length : int
            length of the genotypes
        order : int
            order of epistasis in phenotypes.

        Returns
        -------
        GenotypePhenotypeMap
        """
        wildtype = "0"*length
        mutations = utils.binary_mutations_map(wildtype, "1"*length)
        return cls(wildtype, mutations, **kwargs)

    @classmethod
    def from_coefs(cls, wildtype, mutations, sites, coefs, model_type="global", *args, **kwargs):
        """Construct a genotype-phenotype map from epistatic coefficients.

        Parameters
        ----------
        wildtype : str
            wildtype sequence
        mutations : dict
            dictionary mapping each site to their possible mutations.
        order : int
            order of epistasis
        coefs : list or array
            epistatic coefficients
        model_type : str
            epistatic model to use in composition matrix. (`'global'` or `'local'`)

        Returns
        -------
        GenotypePhenotypeMap
        """
        order = max([len(l) for l in sites])
        self = cls(wildtype, mutations, model_type=model_type, *args, **kwargs)
        if len(coefs) != len(sites):
            raise Exception("""Number of betas does not match order/mutations given.""")
        self.set_coefs(sites, coefs)
        return self

    def build(self, values=None, **kwargs):
        """ Method for construction phenotypes from model. """
        raise Exception("""Must be implemented in subclass. """)

    def set_stdeviations(self, sigma):
        """Add standard deviations to the simulated phenotypes, which can then be
        used for sampling error in the genotype-phenotype map.

        Parameters
        ----------
        sigma : float or array-like
            Adds standard deviations to the phenotypes. If float, all phenotypes
            are given the same stdeviations. Else, array must be same length as
            phenotypes and will be assigned to each phenotype.
        """
        stdeviations = np.ones(len(self.phenotypes)) * sigma
        self.stdeviations = stdeviations
