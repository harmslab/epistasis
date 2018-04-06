Plotting in the epistasis package
=================================

The ``epistasis`` package comes with a few functions to plot epistasis data.

Plotting epistatic coefficients
-------------------------------

The plotting module comes with a default function for plotting epistatic
coefficients. It plots the value of the coefficient as bar graphs, the label as
a box plot (see example below), and signficicance as stars using a t-test.

.. code-block:: python

    from epistasis.models import EpistasisLinearRegression
    from epistasis.pyplot import plot_coefs

    # Fit with a model.
    model = EpistasisLinearRegression.read_json("data.json", order=5)
    model.fit()

    # plot the epistasis coeffs
    sites = model.epistasis.sites
    values = model.epistasis.values
    fig, ax = plot_coefs(sites, values)

.. image:: ../img/coefs.png


Plot nonlinear scale
--------------------

Plot a nonlinear scale using the ``pyplot.nonlinear`` module.

.. code-block:: python

    %matplotlib inline
    import matplotlib.pyplot as plt

    from gpmap.simulate import MountFujiSimulation
    from epistasis.models import EpistasisPowerTransform
    from epistasis.pyplot.nonlinear import plot_power_transform

    gpm = MountFujiSimulation.from_length(4, field_strength=-1, roughness=(-2,2))

    model = EpistasisPowerTransform(lmbda=1, A=0, B=0)
    model.add_gpm(gpm)
    model.fit()

    fig, ax = plt.subplots(figsize=(3,3))

    plot_power_transform(model, cmap='plasma', ax=ax, yerr=0.6)
