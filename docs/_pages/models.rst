Models
======

The following models are available in the `epistasis` API. All models include a
``predict``

Linear
------


EpistasisLinearRegression
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

    def

    model = EpistasisLinearRegression(wildtype, genotypes, phenotypes, func, order=3, model_type="global")
    model.fit()

EpistasisNonlinearRegression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

    import numpy as np

    # Define the nonlinear relationship and it's inverse.
    def exp(x):
        return np.exp(x)

    def log(y):
        return np.log(y)

    model = EpistasisNonlinearRegression(wildtype, genotypes, phenotypes, order=3, model_type="global")
    model.fit(A=1, B=1, lmbda=4)

.. code-block::

    model.fit(A=(0,1,.1), B=(0,1,.1), lmbda=(0,4,.1), use_widgets=True)

Nonlinear
---------

EpistasisPowerTransform
~~~~~~~~~~~~~~~~~~~~~~~



EpistasisNonlinearRegression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classification
--------------
