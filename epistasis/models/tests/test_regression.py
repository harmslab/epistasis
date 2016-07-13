from nose import tools
import numpy as np
from .base import BaseTestClass
from ..regression import EpistasisRegression
from epistasis.mapping import EpistasisMap

class testEpistasisRegression(BaseTestClass):

    def setUp(self):
        super(testEpistasisRegression, self).setUp()
        self.model = EpistasisRegression(
            self.wildtype,
            self.genotypes,
            self.phenotypes,
            order=2,
            stdeviations=self.stdeviations,
            log_transform=self.log_transform,
            n_replicates=self.n_replicates,
            mutations=self.mutations
        )

    def test_init(self):
        """Test initialization of epistasis model."""
        model = EpistasisRegression(
            self.wildtype,
            self.genotypes,
            self.phenotypes,
            order=2,
            stdeviations=self.stdeviations,
            log_transform=self.log_transform,
            n_replicates=self.n_replicates,
        )
        tools.assert_is_instance(model, EpistasisRegression)
        np.testing.assert_equal(model.genotypes, self.genotypes)
        np.testing.assert_equal(model.phenotypes, self.phenotypes)
        tools.assert_true(model.log.transformed)
        tools.assert_is_instance(model.epistasis, EpistasisMap)
        tools.assert_false(model.epistasis.transformed)
        tools.assert_true(model.epistasis.log.transformed)

    def test_fit(self):
        """Test fitting."""
        self.model.fit()
        tools.assert_true(hasattr(self.model, "statistics"))

    def test_predict(self):
        """Test the predictions of the regression model"""
        model = EpistasisRegression(
            self.wildtype,
            self.genotypes,
            self.phenotypes,
            order=4,
            stdeviations=self.stdeviations,
            log_transform=self.log_transform,
            n_replicates=self.n_replicates,
        )
        model.fit()
        np.testing.assert_array_equal(self.genotypes, model.complete_genotypes)
        np.testing.assert_array_equal(model.binary.genotypes, model.binary.complete_genotypes)
        # Check the predicted values!
        np.testing.assert_almost_equal(model.statistics.predict(), self.phenotypes)
