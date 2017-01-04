Basic Examples
==============

This page includes a few things you can do with the epistasis package. To learn
more about modeling epistasis and how to interpret the output, check out our paper
on the topic `here`_.

The following tutorials don't go into too much detail about the model objects
themselves. If you have questions about how to use the package, visit the `gitter page`_.
Otherwise, check out the API docs. 

Fitting with a linear epistasis model
-------------------------------------

.. code-block:: python

    # Import epistasis model
    from epistasis.models import LinearEpistasisModel

    # Read data from file and estimate epistasis
    model = LinearEpistasisModel.from_json("dataset.json")
    model.fit()

    # Estimate the uncertainty in epistatic coefficients
    model.fit_error()


Fitting with a nonlinear epistasis model
----------------------------------------

If analyzing a nonlinear genotype-phenotype map, use `NonlinearEpistasisModel`
(nonlinear least squares regression) to estimate nonlinearity in map.
can be used to estimate the nonlinearity

.. code-block:: python

    # Import the nonlinear epistasis model
    from epistasis.models import NonlinearEpistasisModel

    # Define a nonlinear function to fit the genotype-phenotype map.
    def boxcox(x, lmbda, lmbda2):
        """Fit with a box-cox function to estimate nonlinearity."""
        return ((x-lmbda2)**lmbda - 1 )/lmbda

    # Read data from file and estimate nonlinearity in dataset.
    model = NonlinearEpistasisModel.from_json("dataset.json"
        order=1,
        function=boxcox,
    )

    # Give initial guesses for parameters to aid in convergence (not required).
    model.fit(lmbda=1, A=1, B=2)


.. _here: http://biorxiv.org/content/early/2016/08/30/072256
.. _gitter page: https://gitter.im/home/explore
