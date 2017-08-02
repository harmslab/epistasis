Plotting
========

The ``epistasis`` package comes with a few functions to plot epistasis data.

Coefficients
------------

The plotting module comes with a default function for plotting epistatic
coefficients. It plots the value of the coefficient as bar graphs, the label as
a box plot (see example below), and signficicance as stars using a t-test.

.. code-block:: python

    from epistasis.models import EpistasisLinearRegression
    from epistasis.plots import coefs

    # Fit with a model.
    model = EpistasisLinearRegression.from_json("data.json", order=5)
    model.fit()

    # plot the epistasis coeffs
    labels = model.interactions.labels
    values = model.interactions.values
    fig, ax = coefs(labels, values)

Figure
~~~~~~

.. image:: ../img/coefs.png
