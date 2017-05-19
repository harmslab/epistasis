Estimating uncertainty
======================

The epistasis package includes a ``sampling`` module for estimating uncertainty in
all parameters in a (Non)linear epistasis models. All ``Sampler`` objects create
a database folder with the epistasis model stored inside a pickle file
and an HDF5 file containing samples used to estimate uncertainty.

The module include two types of samplers:

1. BayesianSampler_
2. BootstrapSampler_

Both samplers have the same methods and attributes. They differ in their philosophy
of sampling a model. See the conversation between Frequentism and Bayesianism in this blog_.

.. _blog: http://jakevdp.github.io/blog/2014/03/11/frequentism-and-bayesianism-a-practical-intro/

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

    # Imports
    import matplotlib.pyplot as plt
    import numpy as np
    import corner

    from epistasis.simulate import LinearSimulation
    from epistasis.models import EpistasisLinearRegression
    from epistasis.sampling.bayesian import BayesianSampler

    # Create a simulated genotype-phenotype map with epistasis.
    sim = LinearSimulation.from_length(4, model_type="local")
    sim.set_coefs_order(4)
    sim.set_coefs_random((-1,1))
    sim.set_stdeviations([0.01])

    # Initialize an epistasis model and fit a ML model.
    model = EpistasisLinearRegression.from_gpm(sim, order=4, model_type="local")
    model.fit()

    # Initialize a sampler.
    fitter = BayesianSampler(model)
    fitter.add_samples(500)

    # Plot the Posterior
    fig = corner.corner(bayes.coefs.value, truths=sim.epistasis.values)


.. image:: ../_img/bayes-estimate-uncertainty.png


Defining a prior
~~~~~~~~~~~~~~~~

The default prior for a BayesianSampler is a flat prior (``BayesianSampler.lnprior()``
returns a log-prior equal to 0). To set your own prior, define your own function
that called ``lnprior`` that returns a log prior for a set of `coefs` and reset
the BayesianSampler static method:

.. code-block:: python

    def lnprior(coefs):
        # Set bound on the first coefficient.
        if coefs[0] < 0:
            return -np.inf
        return 0

    # Apply to fitter from above
    fitter.lnprior = lnprior

API
~~~

.. _BayesianSampler:

BayesianSampler
---------------

.. autoclass:: epistasis.sampling.bayesian.BayesianSampler
    :members:
    :inherited-members:

.. _BootstrapSampler:

BootstrapSampler
----------------

.. autoclass:: epistasis.sampling.bootstrap.BootstrapSampler
    :members:
    :inherited-members:
