from ..base import *
import numpy as np
import pytest

class TestBaseSimulation(object):
    
    wildtype = "00"
    mutations = {0:["0","1"], 1:["0","1"]}

    def test_init(self):
        sim = BaseSimulation(self.wildtype, self.mutations)
        # Checks
        assert isinstance(sim, BaseSimulation)
        
    def test_set_coefs_order(self):
        sim = BaseSimulation(self.wildtype, self.mutations)
        sim.set_coefs_order(2)
        # Tests 1: Check that epistasis map is attached
        assert hasattr(sim, "epistasis")
        # Test 2: Check epistasis is built
        assert hasattr(sim.epistasis, "sites")
        # Test 3: Sites have the correct length
        assert len(sim.epistasis.sites) == 4

    def test_set_coefs_sites(self):
        sim = BaseSimulation(self.wildtype, self.mutations)
        sites = [[0], [1], [2]]
        sim.set_coefs_sites(sites)
        # Tests 1: Check that epistasis map is attached
        assert hasattr(sim, "epistasis")
        # Test 2: Check epistasis is built
        assert hasattr(sim.epistasis, "sites")
        # Test 3: Sites have the correct length
        assert len(sim.epistasis.sites) == 3

    def test_build(self):
        sim = BaseSimulation(self.wildtype, self.mutations)
        with pytest.raises(Exception):
                sim.build()
