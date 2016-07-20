from nose import tools
from .base import testBaseClass
from ..base import BaseSimulation

class testBaseSimulation(testBaseClass):

    def test_quick_start(self):
        space = BaseSimulation.quick_start(4,4)
        tools.assert_equal(space.wildtype, "0000")
