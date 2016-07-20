from nose import tools
from .base import testBaseClass
from ..additive import AdditiveSimulation

class testAdditiveSimulation(testBaseClass):

    def test_init(self):
        sim = AdditiveSimulation(
            self.wildtype,
            self.mutations,
            self.order
        )
