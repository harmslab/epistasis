#
# Class for specifying the order of regresssion by given test
#
import json

# imports from seqspace dependency
from seqspace.gpm import GenotypePhenotypeMap
from seqspace.utils import farthest_genotype, binary_mutations_map

from epistasis.stats import log_likelihood_ratio, F_test
from epistasis.models.regression import EpistasisRegression

# -----------------------------------------------------------------------
# Model Specifier Object
# -----------------------------------------------------------------------

class StatisticalTest(object):
    
    def __init__(self, test_type="ftest", cutoff=0.05):
        """ Container for specs on the statistical test used in specifier class below. s"""
        
        # Select the statistical test for specifying model
        test_types = {
            "likelihood": log_likelihood_ratio, 
            "ftest": F_test
        }
        
        # p-value cutoff
        self.cutoff = cutoff
        
        # Set the test type
        self.type = test_type
        
        # Testing function used.
        self.method = test_types[self.type]
        
    @property
    def order(self):
        """ Return the order of the best model"""
        return self.best_model.order
        
    def compare(self, null_model, alt_model):
        """
            Compare two models based on statistic test given
            
            Return the test's statistic value and p-value.
        """
        self.statistic, self.p_value = self.method(null_model, alt_model)
        
        # If test statistic is less than f-statistic cutoff, than keep alternative model
        if self.p_value < self.cutoff:
            self.best_model = alt_model
        # Else, the null model is sufficient and we keep it
        else:
            self.best_model = null_model
            
        return self.statistic, self.p_value

class LinearEpistasisSpecifier:

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
        self.gpm = GenotypePhenotypeMap(
            wildtype,
            genotypes,
            phenotypes,
            log_transform=log_transform,
            mutations=mutations,
            n_replicates=n_replicates        
        )
        
        self.model_type = model_type
        
        # Defaults to binary mapping if not specific mutations are named
        if mutations is None:
            mutant = farthest_genotype(wildtype, genotypes)
            mutations = binary_mutations_map(wildtype, mutant)

        # Testing specs
        self.StatisticalTest = StatisticalTest(test_type, test_cutoff)


    def compare(self, null_order, alt_order):
        """
            Test a higher model against a null model.
            
            This is just a useful utility function for the user... not actually used in this class.
            
            Returns a StatisticalTest object with the best model chosen.
        """
        null_model = EpistasisRegression(wildtype, genotypes, phenotypes, 
            order=null_order, 
            log_transform=log_transform, 
            mutations=mutations, 
            n_replicates=n_replicates, 
            model_type=self.model_type
        )
            
        alt_model =  EpistasisRegression(wildtype, genotypes, phenotypes, 
            order=alt_order, 
            log_transform=log_transform, 
            mutations=mutations, 
            n_replicates=n_replicates, 
            model_type=self.model_type
        )
        
        statistical_test = StatisticalTest(self.test_type, self.StatisticalTest.cutoff)
        return statistical_test.compare(null_model, alt_model)
        
    def fit(self):
        """ Run the specifier method. """
        
        # Begin by starting with a null model. 
        null_model = EpistasisRegression.from_gpm(self.gpm, 
            order=1, 
            model_type=self.model_type
        )
        
        null_model.fit()

        # Construct the range of order
        orders = range(2, self.gpm.length+1)

        self.tests = []

        # Iterate through orders until we reach our significance statistic
        for order in orders:
            # alternative model
            
            alt_model = EpistasisRegression.from_gpm(self.gpm,
                order=order,
                model_type=self.model_type
            )
            alt_model.fit()

            # Initialize statistical test.
            test = StatisticalTest(self.StatisticalTest.type, self.StatisticalTest.cutoff)

            # Run test and append statistic to test_stats
            model_stat, p_value = test.compare(null_model, alt_model)
            self.tests.append(test)
            
            # Set the last test to the most recent.
            self.StatisticalTest = test
            
            # If test statistic is not better than f-statistic cutoff, break loop and choose null
            if p_value < self.StatisticalTest.cutoff:
                null_model = alt_model
            else:
                break
                
    @property
    def p_values(self):
        """ Return list of p-values from all tests in Specifier. """
        return [t.p_value for t in self.tests]
        
    @property
    def scores(self):
        return [t.best_model.Stats.score for t in self.tests[:-1]]


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
        
        
    @classmethod
    def from_json(cls, filename, **kwargs):
        """ Load from json file."""

        # Open, json load, and close a json file
        f = open(filename, "r")
        data = json.load(f)
        f.close()
    
        # Grab all properties from data-structure
        args = ["wildtype", "genotypes", "phenotypes"]
        options = {
            "log_transform": False, 
            "mutations": None,
            "n_replicates": 1,
            "model_type":"local",
            "test_type": "ftest"
        }
    
        # Grab all arguments and order them
        for i in range(len(args)):
            # Get all args
            try: 
                args[i] = data[args[i]]
            except KeyError:
                raise LoadingException("""No `%s` property in json data. Must have %s for GPM construction. """ % (args[i], args[i]) )
    
        # Get all options for map and order them
        for key in options:
            # See if options are in json data
            try:
                options[key] = data[key]
            except:
                pass
    
        # Override any properties with specified kwargs passed directly into method
        options.update(kwargs)
    
        # Create an instance
        specifier = cls(args[0], args[1], args[2], **options)
        return specifier