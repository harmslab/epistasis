"""
NonlinearSimulation
===================

Often, the genotype-phenotype map is nonlinear. That is to say, the genotypes and
phenotypes change on different scales. Genotypes, for example, differ by discrete,
linear changes in sequences (known as mutations). How these changes translate to
phenotype may be less obvious. Sometimes, the effects of mutations simply add together.
Sometimes, the effects multiply. 

The ``epistasis`` package has ``NonlinearSimulation``
class that allows you to construct these more complicated genotype-phenotype maps.
Simply define a function which transforms a linear genotype-phenotype map onto
a nonlinear scale. Note, the function must have ``x`` as the first argument. This
argument represents the linearized phenotypes to be transformed.
"""
import matplotlib.pyplot as plt
import numpy as np

# Import NonlinearSimulation from simulate module
from epistasis.simulate import NonlinearSimulation

###############################################
# Define a nonlinear function that scales the
# the genotype-phenotype map. 
#
# The example below models diminishing returns.

def diminishing_returns(x, A):
    return 1.0 / (1 + A * np.exp(-x))  

x = np.linspace(0,5, 1000)
y = diminishing_returns(x, 2)
plt.plot(x, y)
plt.title("Diminishing returns function")
plt.show()

################################################
# Define the wildtype/ancestor and mutations
# in the genotype-phenotype map. Then, initialize 
# the simulation.

wildtype = "0000"
mutations = {
    0: ["0","1"],
    1: ["0","1"],
    2: ["0","1"],
    3: ["0","1"]
}

# The simulation takes the nonlinear function as an argument
# and the p0 takes values for any extra parametes in the function.

sim = NonlinearSimulation(wildtype, mutations, 
    function=diminishing_returns, 
    p0=[2], 
    model_type="local")

# Set the epistatis coefficients
sim.set_coefs_order(2)
sim.set_coefs_random((-.1,2))

plt.plot(sim.p_additive, sim.phenotypes, 'o')
plt.xlabel("additive phenotype")
plt.ylabel("nonlinear phenotypes")
plt.show()

