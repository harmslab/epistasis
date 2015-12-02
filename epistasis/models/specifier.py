#
# Class for specifying the order of regresssion by given test
#
# imports from seqspace dependency
from seqspace.utils import farthest_genotype, binary_mutations_map

from epistasis.stats import log_likelihood_ratio, F_test
from epistasis.models.regression import EpistasisRegression

# -----------------------------------------------------------------------
# Model Specifier Object
# -----------------------------------------------------------------------

class ModelSpecifier:

    def __init__(self, wildtype, genotypes, phenotypes, test_cutoff=0.05, log_transform=False, mutations=None, n_replicates=1, model_type="local", test_type="ftest"):
        """
        Model specifier. Chooses the order of model based on statistical test.

        """
        # Defaults to binary mapping if not specific mutations are named
        if mutations is None:
            mutant = farthest_genotype(wildtype, genotypes)
            mutations = binary_mutations_map(wildtype, mutant)

        # Select the statistical test for specifying model
        test_types = {"likelihood": log_likelihood_ratio, "ftest": F_test}

        # Testing specs
        self.test_type = test_type
        self.test_method = test_types[test_type]
        self.test_cutoff = test_cutoff

        # Best fit model specs
        self.model_type = model_type
        self.model_order = 1
        self.model_p_value = None
        self.model_stat = None
        self.model = EpistasisRegression(wildtype, genotypes, phenotypes, order=self.model_order, log_transform=log_transform, mutations=mutations, n_replicates=n_replicates, model_type=self.model_type)
        self.model.fit()
        self._specifier()

    def _specifier(self):
        """ Run the specifier method. """
        # Get model specs
        wildtype = self.model.wildtype
        genotypes = self.model.genotypes
        phenotypes = self.model.Raw.phenotypes
        log_transform = self.model.log_transform
        mutations = self.model.mutations
        n_replicates = self.model.n_replicates

        # Construct the range of order
        orders = range(2, len(wildtype)+1)

        self.test_stats = []

        # Iterate through orders until we reach our significance statistic
        for order in orders:
            # alternative model
            alt_model = EpistasisRegression(wildtype, genotypes, phenotypes, order=order, log_transform=log_transform, mutations=mutations, n_replicates=n_replicates, model_type=self.model_type)
            alt_model.fit()

            # Run test and append statistic to test_stats
            model_stat, p_value = self.test_method(self.model, alt_model)
            self.test_stats.append(model_stat)

            # If test statistic is less than f-statistic cutoff, than keep alternative model
            if p_value < self.test_cutoff:
                self.model = alt_model
            # Else, the null model is sufficient and we keep it
            else:
                self.model_order = order-1
                self.model_stat = model_stat
                self.model_p_value = p_value
                break


    # ---------------------------------------------------------------------------------
    # Loading method
    # ---------------------------------------------------------------------------------
        
    @classmethod    
    def from_gpm(cls, gpm, **kwargs):
        """ Initialize an epistasis model from a Genotype-phenotypeMap object """
        
        # Grab un scaled phenotypes and errors
        if gpm.log_transform is True:
            _phenotypes = gpm.Raw.phenotypes
        else:
            _phenotypes = gpm.phenotypes
        
        # Grab each property from map
        model = cls(gpm.wildtype, 
                    gpm.genotypes, 
                    _phenotypes, 
                    mutations = gpm.mutations,
                    log_transform= gpm.log_transform,
                    n_replicates = gpm.n_replicates,
                    **kwargs)
        
        return model
