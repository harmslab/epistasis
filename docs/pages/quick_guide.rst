Quick Guide
===========

Introduction
------------

A Python API for modeling statistical, high-order epistasis in genotype-phenotype maps.
This library provides modules to:

    1. Decompose genotype-phenotype maps into high-order epistatic interactions
    2. Find nonlinear scales in the genotype-phenotype map
    3. Calculate the contributions of different epistatic orders and
    4. Estimate the uncertainty in the epistatic coefficients and

For more information about the epistasis models in this library, see our Genetics paper:

    `Sailer, Z. R., & Harms, M. J. (2017). "Detecting High-Order Epistasis in Nonlinear Genotype-Phenotype Maps." Genetics, 205(3), 1079-1088.`_

.. _`Sailer, Z. R., & Harms, M. J. (2017). "Detecting High-Order Epistasis in Nonlinear Genotype-Phenotype Maps." Genetics, 205(3), 1079-1088.`: http://www.genetics.org/content/205/3/1079


Basic usage
-----------

Import a model from the ``epistasis.models`` module

.. code-block:: python
  
  from epistasis.models import EpistasisLinearRegression
  
Initialize the model, setting the order and type of model (see **Anatomy of the library** for more info).

.. code-block:: python
  
  model = EpistasisLinearRegression(order=3, model_type='global')

Add genotype-phenotype map data. Use the ``gpmap`` library to load such data.

.. code-block:: python
  
  import gpmap
  
  datafile = 'data.csv'
  gpm = gpmap.GenotypePhenotypeMap.read_csv(datafile)
  
  # Add the data.
  model.add_gpm(gpm)
  
Fit the model.

.. code-block:: python
  
  model.fit()


Overview of models
------------------

* **EpistasisLinearRegression**: estimate epistatic coefficents in a linear genotype-phenotype map.
* **EpistasisLasso**: estimate *sparse* epistatic coefficients in a linear genotype-phenotype map
* **EpistasisNonlinearRegression**: estimates high-order epistatic coefficients in a nonlinear genotype-phenotype map.
* **EpistasisNonlinearLasso**: estimate *sparse* epistatic coefficients in a nonlinear genotype-phenotype map.
* **EpistasisPowerTransform**: use a power transform function to fit a nonlinear genotype-phenotype map and estimate epistasis.
* **EpistasisPowerLasso**: use a power transform function to fit a nonlinear genotype-phenotype map and estimate *sparse* epistasis.
* **EpistasisLogisticRegression**: use logistic regression to classify phenotypes as dead/alive.
* **EpistasisMixedRegression**: classify a genotype-phenotype map first, then estimate epistatic coefficients in "alive" phenotypes.

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

.. _gpmap: https: //github.com/harmslab/gpmap
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
