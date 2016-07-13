from nose import tools
import numpy as np
from .base import BaseTestClass
from ..linear import LinearEpistasisModel
from epistasis.mapping import EpistasisMap

class testLinearEpistasisModel(BaseTestClass):

    def setUp(self):
        super(testLinearEpistasisModel, self).setUp()
        self.model = LinearEpistasisModel(
            self.wildtype,
            self.genotypes,
            self.phenotypes,
            stdeviations=self.stdeviations,
            log_transform=self.log_transform,
            n_replicates=self.n_replicates,
            mutations=self.mutations
        )

    def test_init(self):
        """Test initialization of epistasis model."""
        model = LinearEpistasisModel(
            self.wildtype,
            self.genotypes,
            self.phenotypes,
            stdeviations=self.stdeviations,
            log_transform=self.log_transform,
            n_replicates=self.n_replicates,
        )
        tools.assert_is_instance(model, LinearEpistasisModel)
        np.testing.assert_equal(model.genotypes, self.genotypes)
        np.testing.assert_equal(model.phenotypes, self.phenotypes)
        tools.assert_true(model.log.transformed)
        tools.assert_is_instance(model.epistasis, EpistasisMap)
        tools.assert_false(model.epistasis.transformed)
        tools.assert_true(model.epistasis.log.transformed)

    def test_fit(self):
        """Test fitting."""
        self.model.fit()

    def test_fit_error(self):
        """Fit errors."""
        self.model.fit()
        self.model.fit_error()
