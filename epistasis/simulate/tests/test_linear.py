
from ..linear import *
import numpy as np
from nose import tools

def test_LinearSimulation_init():
    wildtype = "00"
    mutations = {0:["0","1"], 1:["0","1"]}
    sim = LinearSimulation(wildtype, mutations)
    # Checks
    check1 = isinstance(sim, LinearSimulation)
    tools.assert_true(check1)
