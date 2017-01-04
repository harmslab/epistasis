Simulating high-order epistasis
===============================

Epistasis can take many forms. It doesn't have to be linear or additive. The
`epistasis` package provides flexibles classes
to make simulating different forms of epistasis easy. A few out-of-box classes
examples are shown below.

Simulating an linear epistatic genotype-phenotype map
-----------------------------------------------------
The following examples show a variety ways to simulate a genotype-phenotype map
with linear, high-order epistatic interactions. The simulation interface provides
methods to easily dictate the construction of a simulated genotype-phenotype map.

```python
from epistasis.simulate import AdditiveSimulation

# Define the wildtype sequence and possible mutations at each site.
wildtype = "0000"
mutations = {
    0: ["0", "1"], # mutation alphabet for site 0
    1: ["0", "1"],
    2: ["0", "1"],
    3: ["0", "1"]
}
# Initialize a simulation
gpm = AdditiveSimulation(wildtype, mutations)

# Set the order of epistasis
gpm.set_coefs_order(4)

# Generate random epistatic coefficients
coef_range = (-1, 1) # coefs between -1 and 1
gpm.set_coefs_random(coef_range)
```

Alternatively, you can quickly simulate a binary genotype-phenotype map if you're
fine with a simple, binary alphabet at each site.
```python
# define the length of genotypes and the order of epistasis
length = 4
gpm = AdditiveSimulation.from_length(length)

# Generate random epistatic coefs
gpm.set_coefs_order(4)
gpm.set_coefs_random(coef_range)
```

For all simulated genotype-phenotype maps, one can initialize a genotype-phenotype
map from an existing dataset. Scroll through class methods that start with `from_` to
see all options for initializing simulated genotype-phenotype maps.

Simulating a multiplicative epistatic genotype-phenotype map
------------------------------------------------------------
