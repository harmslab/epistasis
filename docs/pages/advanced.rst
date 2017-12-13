Advanced topics
===============

Setting bounds on nonlinear fits
--------------------------------

All nonlinear epistasis models use lmfit_ to estimate a nonlinear scale in an
arbitrary genotype-phenotype map. Each model creates ``lmfit.Parameter`` objects
for each parameter in the input function and contains them in the ``parameters``
attribute as ``lmfit.Parameters`` object. Thus, you can set the bounds, initial
guesses, etc. on the parameters following lmfit's API. The model, then, minimizes
the squared residuals using the ``lmfit.minimize`` function. The results are
stored in the ``Nonlinear`` object.

In the example below, we use a ``EpistasisPowerTransform`` to demonstrate how to
access the lmfit API.


.. _lmfit: https://lmfit.github.io/lmfit-py/

.. code-block:: python

    # Import a nonlinear model (this case, Power transform)
    from epistasis.models import NonlinearPowerTransform

    model = NonlinearPowerTransform(order=1)
    model.parameters['lmbda'].set(value=1, min=0, max=10)
    model.parameters['A'].set(value=10, min=0, max=100)

    model.fit()


Access information about the minimizer results using the ``Nonlinear`` attribute.

.. code-block:: python

    # Pretty print the results!
    model.Nonlinear.params.pretty_print()


Estimating model uncertainty
----------------------------

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
