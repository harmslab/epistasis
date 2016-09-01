#
# Class for specifying the order of regresssion by given test
#
import json
import numpy as np

# imports from seqspace dependency
from seqspace.gpm import GenotypePhenotypeMap
from seqspace.utils import (farthest_genotype,
                            binary_mutations_map)

from epistasis.stats import log_likelihood_ratio, F_test, StatisticalTest
from epistasis.utils import SubclassException

# Linear Regressionlinear
from epistasis.models.regression import LinearEpistasisRegression

# Nonlinear Regression
from epistasis.models.nonlinear import NonlinearEpistasisModel

# -----------------------------------------------------------------------
# Model Specifier Object
# -----------------------------------------------------------------------

class BaseSpecifier(object):

    def __init__(self, wildtype, genotypes, phenotypes,
        test_cutoff=0.05,
        log_transform=False,
        mutations=None,
        n_replicates=1,
        model_type="local",
        test_type="ftest",
        logbase=np.log10):
        self.gpm = GenotypePhenotypeMap(
            wildtype,
            genotypes,
            phenotypes,
            log_transform=log_transform,
            mutations=mutations,
            n_replicates=n_replicates,
            logbase=logbase
        )

        self.model_type = model_type
        # Defaults to binary mapping if not specific mutations are named
        if mutations is None:
            mutant = farthest_genotype(wildtype, genotypes)
            mutations = binary_mutations_map(wildtype, mutant)

        # Testing specs
        self.StatisticalTest = StatisticalTest(test_type, test_cutoff)

    def compare(self, null_order, alt_order):
        """ Compare two models with different orders. """
        raise SubclassException("""Must be implemented in a subclass.""")

    def fit(self):
        """ Fit a specifier to data. """
        raise SubclassException("""Must be implemented in a subclass.""")

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
            _phenotypes = gpm.phenotypes
        else:
            _phenotypes = gpm.log.phenotypes

        # Grab each property from map
        model = cls(gpm.wildtype,
                    gpm.genotypes,
                    _phenotypes,
                    mutations = gpm.mutations,
                    log_transform= gpm.log_transform,
                    n_replicates = gpm.n_replicates,
                    logbase=gpm.logbase
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
            "test_type": "ftest",
            "logbase":np.log10
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

class LinearEpistasisSpecifier(BaseSpecifier):

    """Model specifier. Chooses the order of model based on any statistical test.

    On initialization, this class automatically finds the appropriate model
    based on cutoffs and test type. Default statistical test is F-test.
    """
    def __init__(self, wildtype, genotypes, phenotypes,
        test_cutoff=0.05,
        log_transform=False,
        mutations=None,
        n_replicates=1,
        model_type="local",
        test_type="ftest"):

        # Inherit base class init
        super(LinearEpistasisSpecifier,self).__init__(wildtype, genotypes, phenotypes,
            test_cutoff=0.05,
            log_transform=False,
            mutations=mutations,
            n_replicates=1,
            model_type=model_type,
            test_type=test_type)

    def compare(self, null_order, alt_order):
        """Test a higher model against a null model.

        This is just a useful utility function for the user... not actually used in this class.
        Returns a StatisticalTest object with the best model chosen.
        """
        null_model = LinearEpistasisRegression.from_gpm(
            self.gpm,
            order=null_order,
            model_type=self.model_type
        )
        null_model.fit()


        alt_model = LinearEpistasisRegression.from_gpm(
            self.gpm,
            order=alt_order,
            model_type=self.model_type
        )
        alt_model.fit()

        statistical_test = StatisticalTest(self.StatisticalTest.type, self.StatisticalTest.cutoff)
        return statistical_test.compare(null_model, alt_model)

    def fit(self):
        """ Run the specifier method. """

        # Begin by starting with a null model.
        null_model = LinearEpistasisRegression.from_gpm(self.gpm,
            order=0,
            model_type=self.model_type
        )

        null_model.fit()

        # Construct the range of order
        orders = range(1, self.gpm.length+1)

        self.tests = []

        # Iterate through orders until we reach our significance statistic
        for order in orders:
            # alternative model

            alt_model = LinearEpistasisRegression.from_gpm(self.gpm,
                order=order,
                model_type=self.model_type
            )
            alt_model.fit()

            # Initialize statistical test.
            test = StatisticalTest(self.StatisticalTest.type, self.StatisticalTest.cutoff)

            # Run test and append statistic to test_stats
            model_stat, p_value = test.compare(null_model, alt_model)
            self.tests.append(test)

            # If test statistic is not better than f-statistic cutoff, break loop and choose null
            if p_value < self.StatisticalTest.cutoff:
                null_model = alt_model

                # Set the last test to the most recent.
                self.StatisticalTest = test
            else:
                break


class NonlinearEpistasisSpecifier(BaseSpecifier):
    """Model specifier. Chooses the order of model based on any statistical test.

    On initialization, this class automatically finds the appropriate model
    based on cutoffs and test type.

    Default statistical test is F-test.

    """
    def __init__(self, wildtype, genotypes, phenotypes, function,
        test_cutoff=0.05,
        log_transform=False,
        mutations=None,
        n_replicates=1,
        model_type="local",
        test_type="ftest"):


        # Inherit base class init
        super(NonlinearEpistasisSpecifier,self).__init__(wildtype, genotypes, phenotypes,
            test_cutoff=0.05,
            log_transform=False,
            mutations=mutations,
            n_replicates=1,
            model_type=model_type,
            test_type=test_type)

        self.function = function


    def compare(self, null_order, alt_order):
        """Test a higher model against a null model.

        This is just a useful utility function for the user... not actually used in this class.

        Returns a StatisticalTest object with the best model chosen.
        """
        null_model = NonlinearEpistasisModel.from_gpm(
            self.gpm,
            function = self.function,
            order=null_order,
            model_type=self.model_type
        )
        null_model.fit()


        alt_model = NonlinearEpistasisModel.from_gpm(
            self.gpm,
            function=self.function,
            order=alt_order,
            model_type=self.model_type
        )
        alt_model.fit()

        statistical_test = StatisticalTest(self.StatisticalTest.type, self.StatisticalTest.cutoff)
        return statistical_test.compare(null_model, alt_model)

    def fit(self):
        """ Run the specifier method. """

        # Begin by starting with a null model.
        null_model = NonlinearEpistasisModel.from_gpm(self.gpm,
            function=self.function,
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

            alt_model = NonlinearEpistasisModel.from_gpm(self.gpm,
                function=self.function,
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

                # Set the last test to the most recent.
                self.StatisticalTest = test
            else:
                break
