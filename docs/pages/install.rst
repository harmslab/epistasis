Installation
============

Installing
----------

This library is now available on PyPi, so it can be installed using pip.

.. code-block:: bash

    pip install epistasis

For the latest version of the package, you can also clone from Github
and install a development version using pip.

.. code-block:: bash

    git clone https://github.com/harmslab/epistasis
    cd epistasis
    pip install -e .


Dependencies
------------

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

Testing
-------

The epistasis package comes with a suite of tests. Running the tests require `pytest`, 
so make sure it is installed.

.. code-block:: bash

    pip install -U pytest

Once pytest is installed, run the tests from the base directory of the epistasis package
using the following command.

.. code-block:: bash

    pytest
