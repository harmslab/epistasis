Simulating
==========

Epistasis can take many forms. It can be nonlinear and/or high-order (i.e. higher
than three-way interactions). The ``epistasis`` package provides flexibles classes
build simulated genotype-phenotype maps that exhibit such features. A few out-of-box classes
examples are shown below.

Basic Linear Example
--------------------

The following examples show a variety ways to simulate a genotype-phenotype map
with linear, high-order epistatic interactions. The simulation interface provides
methods to easily dictate the construction of a simulated genotype-phenotype map.

.. code-block:: python

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

Alternatively, you can quickly simulate a binary genotype-phenotype map if you're
fine with a simple, binary alphabet at each site.

.. code-block:: python

    # define the length of genotypes and the order of epistasis
    length = 4
    gpm = AdditiveSimulation.from_length(length)

    # Generate random epistatic coefs
    gpm.set_coefs_order(4)
    gpm.set_coefs_random(coef_range)

For all simulated genotype-phenotype maps, one can initialize a genotype-phenotype
map from an existing dataset. Scroll through class methods that start with ``from_`` to
see all options for initializing simulated genotype-phenotype maps.

Basic Nonlinear Example
-----------------------

Often, the genotype-phenotype map is nonlinear. That is to say, the genotypes and
phenotypes change on different scales. Genotypes, for example, differ by discrete,
linear changes in sequences (known as mutations). How these changes translate to
phenotype may be less obvious. Sometimes, the effects of mutations simply add together.
Sometimes, the effects multiply. The ``epistasis`` package has ``NonlinearSimulation``
class that allows you to construct these more complicated genotype-phenotype maps.
Simply define a function which transforms a linear genotype-phenotype map onto
a nonlinear scale. Note, the function must have ``x`` as the first argument. This
argument represents the linearized phenotypes to be transformed.

.. code-block:: python

    from epistasis.simulate import NonlinearSimulation

    def saturating_scale(x, K):
        return ((K+1)*x)/(K+x)

    # Define the initial value for the paramter, K
    p0 = [2]

    gpm = NonlinearSimulation.from_length(4, function=saturating_scale, p0=p0)
    gpm.set_coefs_order(4)
    gpm.set_coefs_random((0,1))


Basic Multiplicative Example
----------------------------

Multiplicative epistasis is a common nonlinear, phenotypic scale. You can simulate
a multiplicative genotype-phenotype map using the ``NonlinearSimulation`` class.
Remember, the ``epistasis`` package always constructs/decomposes epistasis as a sum of epistatic
coefficients. To construct a multiplicative map, then, simply use the exponential
logarithm of the epistatic coefficients:

.. math::

    \begin{eqnarray}
    p & = & \beta_1 \beta_2 \beta_{1,2} \\
    p & = & e^{ln(\beta_1) + ln(\beta_2) + ln(\beta_{1,2})}
    \end{eqnarray}

Using the ``epistasis`` package, this looks like the following example. First, define
the exponential function as the nonlinear scale passed into the Simulation class.

.. code-block:: python

    import numpy as np
    from epistasis.simulation import NonlinearSimulation

    def multiplicative(x):
        return np.exp(x)

    gpm = NonlinearSimulation.from_length(4, function=multiplicative)

Then, define the epistatic coefficients, take their log, and pass them into the
simulation object.

.. code-block:: python

    # Set the order of epistasis
    gpm.set_coefs_order(4)

    # generate random coefs
    coefs = np.random.uniform(0,3, size=len(gpm.epistasis.labels))

    # Take the log of the coefs
    log_coefs = np.log(coefs)

    # Pass coefs into the simulation class.
    gpm.set_coefs_values(log_coefs)
