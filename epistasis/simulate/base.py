import numpy as np
from gpmap.gpm import GenotypePhenotypeMap
from gpmap import utils

# Local imports
from epistasis.mapping import (EpistasisMap, mutations_to_sites, assert_epistasis)
from epistasis.matrix import get_model_matrix
from epistasis.utils import extract_mutations_from_genotypes
from epistasis.models.utils import XMatrixException


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
                 model_type="global",
                 **kwargs
                 ):
        self.model_type = model_type
        self.Xbuilt = {}
        genotypes = np.array(
            utils.mutations_to_genotypes(wildtype, mutations))
        phenotypes = np.ones(len(genotypes))
        # Initialize a genotype-phenotype map
        super(BaseSimulation, self).__init__(
            wildtype,
            genotypes,
            phenotypes,
            mutations=mutations,
            **kwargs)

    def add_epistasis(self):
        """Add an EpistasisMap to model.
        """
        # Build epistasis interactions as columns in X matrix.
        sites = mutations_to_sites(self.order, self.mutations)

        # Map those columns to epistastalis dataframe.
        self.epistasis = EpistasisMap(
            sites, order=self.order, model_type=self.model_type)

    def add_X(self, X="complete", key=None):
        """Add X to Xbuilt

        Parameters
        ----------
        X :
            see above for details.
        key : str
            name for storing the matrix.

        Returns
        -------
        X_builts : numpy.ndarray
            newly built 2d array matrix
        """
        if type(X) is str and X in ['obs', 'complete']:

            # Create a list of epistatic interaction for this model.
            if hasattr(self, "epistasis"):
                columns = self.epistasis.sites
            else:
                self.add_epistasis()
                columns = self.epistasis.sites

            # Use desired set of genotypes for rows in X matrix.
            index = self.binary

            # Build numpy array
            x = get_model_matrix(index, columns, model_type=self.model_type)

            # Set matrix with given key.
            if key is None:
                key = X

            self.Xbuilt[key] = x

        elif type(X) == np.ndarray or type(X) == pd.DataFrame:
            # Set key
            if key is None:
                raise Exception("A key must be given to store.")

            # Store Xmatrix.
            self.Xbuilt[key] = X

        else:
            raise XMatrixException("X must be one of the following: 'obs',"
                                   " 'complete', "
                                   "numpy.ndarray, or pandas.DataFrame.")

        X_built = self.Xbuilt[key]
        return X_built

    def set_coefs_order(self, order):
        """Construct a set of epistatic coefficients given the epistatic
        order."""
        # Attach an epistasis model.
        self.order = order
        self.add_epistasis()
        self.epistasis.data.values = np.zeros(self.epistasis.n)
        self.epistasis.data.values[0] = 1
        return self

    def set_coefs_sites(self, sites):
        """Construct a set of epistatic coefficients given a list of
        coefficient sites."""
        self.order = max([len(s) for s in sites])
        self.add_epistasis()
        return self

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
        self.epistasis.data.values = values
        self.build()
        return self

    @assert_epistasis
    def set_wildtype_phenotype(self, value):
        """Set the wildtype phenotype."""
        self.epistasis.data.values[0] = value
        self.build()

    @assert_epistasis
    def set_coefs_values(self, values):
        """Set coefficient values.
        """
        self.epistasis.data.values = values
        self.build()
        return self

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
        self.epistasis.data.values = np.random.uniform(
            coef_range[0], coef_range[1], size=len(self.epistasis.sites))
        self.build()
        return self

    @assert_epistasis
    def set_coefs_decay(self):
        """Use a decaying exponential model to choose random epistatic coefficient
        values that decay/shrink with increasing order.
        """
        wt_phenotype = self.epistasis.values[0]
        for order in range(1, self.epistasis.order + 1):
            # Get epistasis map for this order.
            em = self.epistasis.get_orders(order)
            index = em.index

            # Randomly choose values for the given order
            vals = 10**(-order) * np.random.uniform(-wt_phenotype,
                                                    wt_phenotype,
                                                    size=len(index))

            # Map to epistasis object.
            self.epistasis.data.values[index[0]: index[-1] + 1] = vals
        self.build()
        return self

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
        wildtype = "0" * length
        mutations = utils.binary_mutations_map(wildtype, "1" * length)
        return cls(wildtype, mutations, **kwargs)

    @classmethod
    def from_coefs(cls, wildtype, mutations, sites, coefs, model_type="global",
                   *args, **kwargs):
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
            epistatic model to use in composition matrix.
            (`'global'` or `'local'`)

        Returns
        -------
        GenotypePhenotypeMap
        """
        order = max([len(l) for l in sites])
        self = cls(wildtype, mutations,
                   model_type=model_type, *args, **kwargs)
        if len(coefs) != len(sites):
            raise Exception(
                """Number of betas does not match order/mutations given.""")
        self.set_coefs(sites, coefs)
        return self

    def build(self, values=None, **kwargs):
        """ Method for construction phenotypes from model. """
        raise Exception("""Must be implemented in subclass. """)

    def set_stdeviations(self, sigma):
        """Add standard deviations to the simulated phenotypes, which can then
        be used for sampling error in the genotype-phenotype map.

        Parameters
        ----------
        sigma : float or array-like
            Adds standard deviations to the phenotypes. If float, all
            phenotypes are given the same stdeviations. Else, array must be
            same length as phenotypes and will be assigned to each phenotype.
        """
        stdeviations = np.ones(len(self.phenotypes)) * sigma
        self.data['stdeviations'] = stdeviations
        return self
