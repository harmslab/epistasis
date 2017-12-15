Quick Guide
===========

Introduction
------------

``epistasis`` is a Python library that includes models to estimate statistical, high-order epistasis in genotype-phenotype maps. Using this library, you can

    1. Decompose genotype-phenotype maps into high-order epistatic interactions
    2. Find nonlinear scales in the genotype-phenotype map
    3. Calculate the contributions of different epistatic orders and
    4. Estimate the uncertainty in the epistatic coefficients and

For more information about the epistasis models in this library, see our Genetics paper:

    `Sailer, Z. R., & Harms, M. J. (2017). "Detecting High-Order Epistasis in Nonlinear Genotype-Phenotype Maps." Genetics, 205(3), 1079-1088.`_


Simple Tutorial
--------------

Follow these five steps for all epistasis models in this library:

1. **Import a model.** There many models available in the ``epistasis.models`` module. See the full list in the next section.

.. code-block:: python

  from epistasis.models import EpistasisLinearRegression

2. **Initialize a model**. Set the order, choose the type of model (see `Anatomy of an epistasis model`_ for more info), and set any other parameters in the model.

.. code-block:: python

  model = EpistasisLinearRegression(order=3, model_type='global')

3. **Add some data**. There are three basic ways to do this. 1. Pass data directly to the epistasis model using the ``add_data`` method. 2. Read data from a separate file using one of the ``read_`` methods. 3. (The best option.) load data into a GenotypePhenotypeMap object from the GPMap library and add it to the epistasis model.

.. code-block:: python

  from gpmap import GenotypePhenotypeMap

  datafile = 'data.csv'
  gpm = GenotypePhenotypeMap.read_csv(datafile)

  # Add the data.
  model.add_gpm(gpm)

  # model now has a `gpm` attribute.

4. **Fit the model.** Each model has a simple fit method. Call this to estimate epistatic coefficients. The results are stored the ``epistasis`` attribute.

.. code-block:: python

  # Call fit method
  model.fit()

  # model now has an ``epistasis`` attribute

5. **Plot the results.** The epistasis library has a ``pyplot`` module (powered by matplotlib) with a few builtin plotting functions.

.. code-block:: python

  from epistasis.pyplot import plot_coefs

  fig, ax = plot_coefs(model.epistasis.sites, model.epistasis.values)

.. image:: ../img/basic-example.png


Overview of available models
----------------------------

* EpistasisLinearRegression_: estimate epistatic coefficents in a linear genotype-phenotype map.
* EpistasisLasso_: estimate *sparse* epistatic coefficients in a linear genotype-phenotype map
* EpistasisNonlinearRegression_: estimates high-order epistatic coefficients in a nonlinear genotype-phenotype map.
* EpistasisNonlinearLasso_: estimate *sparse* epistatic coefficients in a nonlinear genotype-phenotype map.
* EpistasisPowerTransform_: use a power transform function to fit a nonlinear genotype-phenotype map and estimate epistasis.
* EpistasisPowerLasso_: use a power transform function to fit a nonlinear genotype-phenotype map and estimate *sparse* epistasis.
* EpistasisLogisticRegression_: use logistic regression to classify phenotypes as dead/alive.
* EpistasisMixedRegression_: classify a genotype-phenotype map first, then estimate epistatic coefficients in "alive" phenotypes.

Installation and dependencies
------------------------------

For users
~~~~~~~~~

This library is now available on PyPi, so it can be installed using pip.

.. code-block:: bash

    pip install epistasis

For developers
~~~~~~~~~~~~~~

For the latest version of the package, you can also clone from Github
and install a development version using pip.

.. code-block:: bash

    git clone https://github.com/harmslab/epistasis
    cd epistasis
    pip install -e .


Dependencies
~~~~~~~~~~~~

The following dependencies are required for the epistasis package.

* gpmap_: Module for constructing powerful genotype-phenotype map python data-structures.
* Scikit-learn_: Simple to use machine-learning API.
* Numpy_: Python's array manipulation package.
* Scipy_: Efficient scientific array manipulations and fitting.
* Pandas_: High-performance, easy-to-use data structures and data analysis tools.

There are also some additional dependencies for extra features included in
the package.

* matplotlib_: Python plotting API.
* ipython_: interactive python kernel.
* `jupyter notebook`_: interactive notebook application for running python kernels interactively.
* ipywidgets_: interactive widgets in python.

.. _gpmap: https://github.com/harmslab/gpmap
.. _Scikit-learn: http://scikit-learn.org/stable/
.. _Numpy: http://www.numpy.org/
.. _Scipy: http://www.scipy.org/
.. _Pandas: http://pandas.pydata.org/
.. _matplotlib: http://matplotlib.org/
.. _ipython: https://ipython.org/
.. _jupyter notebook: http://jupyter.org/
.. _ipywidgets: https://ipywidgets.readthedocs.io/en/latest/

Running tests
-------------

The epistasis package comes with a suite of tests. Running the tests require `pytest`,
so make sure it is installed.

.. code-block:: bash

    pip install -U pytest

Once pytest is installed, run the tests from the base directory of the epistasis package
using the following command.

.. code-block:: bash

    pytest

.. Links for this page

.. _`Sailer, Z. R., & Harms, M. J. (2017). "Detecting High-Order Epistasis in Nonlinear Genotype-Phenotype Maps." Genetics, 205(3), 1079-1088.`: http://www.genetics.org/content/205/3/1079
.. _`Anatomy of an epistasis model`: anatomy.html
.. _EpistasisLinearRegression: models.html#epistasislinearregression
.. _EpistasisLasso: models.html#epistasislasso
.. _EpistasisNonlinearRegression: models.html#epistasisnonlinearregression
.. _EpistasisNonlinearLasso: models.html#epistasisnonlinearlasso
.. _EpistasisPowerTransform: models.html#epistasispowertransform
.. _EpistasisPowerLasso: models.html#epistasispowerlasso
.. _EpistasisLogisticRegression: models.html#epistasislogisticregression
.. _EpistasisMixedRegression: models.html#epistasismixedregression
