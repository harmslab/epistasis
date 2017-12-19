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


Plot classification
-------------------
