Plotting and visualizing epistasis
====================================

The ``epistasis`` package comes with a few functions to plot epistasis data.

coefs
-----

The plotting module comes with a default function for plotting epistatic
coefficients. It plots the value of the coefficient as bar graphs, the label as
a box plot (see example below), and signficicance as stars using a t-test.

.. code-block:: python

    from epistasis.models import EpistasisLinearRegression
    from epistasis.plots import coefs

    # Fit with a model.
    model = EpistasisLinearRegression.read_json("data.json", order=5)
    model.fit()

    # plot the epistasis coeffs
    sites = model.epistasis.sites
    values = model.epistasis.values
    fig, ax = coefs(sites, values)

.. image:: ../img/coefs.png


Plot nonlinear scale
--------------------


Plot classification
-------------------
