# import numpy as np
# import pytest
#
# from ..power import PowerScaleSimulation
#
# class TestPowerScaleSimulation(object):
#
#     wildtype = "00"
#     mutations = {0:["0","1"], 1:["0","1"]}
#     p0 = [1,1,1]
#
#     def test_init(self):
#         sim = PowerScaleSimulation(self.wildtype, self.mutations, p0=self.p0)
#         # Checks
#         assert isinstance(sim, PowerScaleSimulation)
#         assert hasattr(sim, "Xbuilt")
#
#     def test_function(self):
#         x = np.random.uniform(1,5, size=10)
#
#         # Run function, catch output
#         output = PowerScaleSimulation.function(x, *self.p0)
#
#         # Assert callable
#         assert callable(PowerScaleSimulation.function)
#         # assert type(output) = np.ndarray
#         assert len(output) == len(x)
#
#     def test_build(self):
#         sim = PowerScaleSimulation(self.wildtype, self.mutations, p0=self.p0)
#         sim.set_coefs_order(2)
#         sim.set_coefs_decay()
#
#         assert hasattr(sim, "phenotypes")
#         assert hasattr(sim, "linear_phenotypes")
#         assert "complete" in sim.Xbuilt
