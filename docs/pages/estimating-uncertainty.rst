Estimating uncertainty in a model
=================================

The epistasis package includes a ``sampling`` module for estimating uncertainty in
all coefficients in (Non)linear epistasis models. It follows a Bayesian approach, 
and uses the `emcee` python package to approximate the posterior distributions 
for each coefficient. The plot below was created using the `corner` package. 

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
    samples = fitter.sample(500)
    
    # Plot the Posterior
    fig = corner.corner(samples, truths=sim.epistasis.values)


.. image:: ../img/bayes-estimate-uncertainty.png


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
