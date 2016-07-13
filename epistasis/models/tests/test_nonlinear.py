from nose import tools
import numpy as np
from .base import BaseTestClass
from ..nonlinear import NonlinearEpistasisModel
from epistasis.mapping import EpistasisMap

class testNonlinearEpistasisModel(BaseTestClass):

    def setUp(self):
        def test_func(x, a, b):
            return a * x + b

        self.function = test_func

        super(testNonlinearEpistasisModel, self).setUp()
        self.model = NonlinearEpistasisModel(
            self.wildtype,
            self.genotypes,
            self.phenotypes,
            self.function,
            order=1,
            stdeviations=self.stdeviations,
            log_transform=self.log_transform,
            n_replicates=self.n_replicates,
            mutations=self.mutations
        )

        self.model.fit(a=1,b=0)

    def test_init(self):
        """Test initialization of epistasis model."""
        tools.assert_is_instance(self.model, NonlinearEpistasisModel)
        np.testing.assert_equal(self.model.genotypes, self.genotypes)
        np.testing.assert_equal(self.model.phenotypes, self.phenotypes)
        tools.assert_is_instance(self.model.epistasis, EpistasisMap)
        tools.assert_true(self.model.linear.log.transformed)

    def test_fit(self):
        """Test fitting."""
        tools.assert_false(self.model.epistasis.transformed)

    def test_parameters(self):
        """Fit errors."""
        tools.assert_true(hasattr(self.model, "parameters"))
        tools.assert_true(hasattr(self.model.parameters, "a"))
        tools.assert_true(hasattr(self.model.parameters, "b"))
