# 
# from ..linear import *
# import numpy as np
#
#
# class TestLinearSimulation(object):
#
#     wildtype = "00"
#     mutations = {0: ["0", "1"], 1: ["0", "1"]}
#
#     def test_init(self):
#         sim = LinearSimulation(self.wildtype, self.mutations)
#         # Checks
#         assert isinstance(sim, LinearSimulation)
#
#     def test_set_coefs(self):
#         sim = LinearSimulation(self.wildtype, self.mutations)
#         sites = [[0], [1], [2]]
#         values = [0.5, 0.2, 0.4]
#         sim.set_coefs(sites, values)
#
#         # Test the correct attributes are set
#         assert hasattr(sim, "epistasis")
#         assert hasattr(sim.epistasis, "_values")
#         assert hasattr(sim.epistasis, "_sites")
#         assert sim.epistasis.values == values
