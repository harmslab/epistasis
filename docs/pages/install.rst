Installation
============

Install
-------

The epistasis package is not yet on Pypi. Our first official release will happen
as soon as our paper is out of preprint. Until then, all our software is available
on Github. You can clone from there and pip install a development version.

.. code-block:: bash

    git clone https://github.com/harmslab/epistasis
    cd epistasis
    pip install -e .


Dependencies
------------

The following dependencies are required for the epistasis package. **Note**:
The `gpmap` package is not yet on Pypi either. It will need to be downloaded
and installed following the same procedure as above.

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
