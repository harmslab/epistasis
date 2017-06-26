
from ..base import *
import numpy as np
from nose import tools

def test_BaseSimulation_init():
    wildtype = "00"
    mutations = {0:["0","1"], 1:["0","1"]}
    sim = BaseSimulation(wildtype, mutations)
    # Checks
    check1 = isinstance(sim, BaseSimulation)
    tools.assert_true(check1)

def test_BaseSimulation_set_coefs_order():
    wildtype = "00"
    mutations = {0:["0","1"], 1:["0","1"]}
    sim = BaseSimulation(wildtype, mutations)
    sim.set_coefs_order(2)
    # Tests 1: Check that epistasis map is attached
    check1 = hasattr(sim, "epistasis")
    tools.assert_true(check1)
    # Test 2: Check epistasis is built
    check2 = hasattr(sim.epistasis, "sites")
    tools.assert_true(check2)
    # Test 3: Sites have the correct length
    check3 = len(sim.epistasis.sites)
    tools.assert_equal(check3, 4)

def test_BaseSimulation_set_coefs_sites():
    wildtype = "00"
    mutations = {0:["0","1"], 1:["0","1"]}
    sim = BaseSimulation(wildtype, mutations)
    sites = [[0], [1], [2]]
    sim.set_coefs_sites(sites)
    # Tests 1: Check that epistasis map is attached
    check1 = hasattr(sim, "epistasis")
    tools.assert_true(check1)
    # Test 2: Check epistasis is built
    check2 = hasattr(sim.epistasis, "sites")
    tools.assert_true(check2)
    # Test 3: Sites have the correct length
    check3 = len(sim.epistasis.sites)
    tools.assert_equal(check3, 3)

def test_BaseSimulation_build_raises_subclass_error():
    wildtype = "00"
    mutations = {0:["0","1"], 1:["0","1"]}
    sim = BaseSimulation(wildtype, mutations)
    tools.assert_raises(Exception, sim.build)
