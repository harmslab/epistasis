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

class BestModel(object):
    
    def __init__(self, model):
        """ Container for statistics for specifier class. """
        self.model = model
        self.p_value = 0
        self.stat = 0
    
    @property
    def score(self):
        return self._model.score
        
    @property
    def order(self):
        return self._model.order

class StatisticTest(object):
    
    def __init__(self, test_type, cutoff):
        """ Container for specs on the statistical test used in specifier class below. s"""
        
        # Select the statistical test for specifying model
        test_types = {
            "likelihood": log_likelihood_ratio, 
            "ftest": F_test
        }
        
        self.cutoff = cutoff
        self.type = test_type
        self.method = test_types[self.type]

class ModelSpecifier:

    def __init__(self, wildtype, genotypes, phenotypes, 
        test_cutoff=0.05, 
        log_transform=False, 
        mutations=None, 
        n_replicates=1, 
        model_type="local", 
        test_type="ftest"):
        
        """
            Model specifier. Chooses the order of model based on any statistical test.
            
            On initialization, this class automatically finds the appropriate model
            based on cutoffs and test type. 
    
            Default statistical test is F-test. 
            
        """
        # Defaults to binary mapping if not specific mutations are named
        if mutations is None:
            mutant = farthest_genotype(wildtype, genotypes)
            mutations = binary_mutations_map(wildtype, mutant)

        # Testing specs
        self.Stats = StatisticTest(test_type, test_cutoff)

        # Best fit model specs
        self.model_type = model_type
        self.model_order = 1
        self.model_p_value = None
        self.model_stat = None


    def compare_models(self, null_order, test_order):
        """
            Test a higher model against a null model.
        """
        null_model = EpistasisRegression(wildtype, genotypes, phenotypes, 
            order=null_order, 
            log_transform=log_transform, 
            mutations=mutations, 
            n_replicates=n_replicates, 
            model_type=self.model_type
        )
            

    def fit(self):
        """ Run the specifier method. """
        # Construct a regression of the data
        self.model = EpistasisRegression(wildtype, genotypes, phenotypes, 
            order=self.model_order, 
            log_transform=log_transform, 
            mutations=mutations, 
            n_replicates=n_replicates, 
            model_type=self.model_type)
            
        # Fit the regression and specify the proper order using test statistic.
        self.model.fit()
        
        # Get model specs
        wildtype = self.model.wildtype
        genotypes = self.model.genotypes
        
        if self.model.log_transform is True:
            phenotypes = self.model.Raw.phenotypes
        else:
            phenotypes = self.model.phenotypes    
        
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