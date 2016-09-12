.. epistasis documentation master file, created by
   sphinx-quickstart on Thu Jul  7 15:47:18 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Epistasis Package Documentation
===============================

A Python API for modeling statistical, high-order epistasis in large
genotype-phenotype maps. Decompose genotype-phenotype maps into high-order epistatic
interactions. Find nonlinearity in the genotype-phenotype map. Calculate the
contributions of different epistatic orders. Estimate the importance of high-order
interactions on evolution.

Currently, this package works only as an API. There is not a command-line
interface, and it includes few ways to read/write the data to disk out-of-the-box.
We plan to improve this moving forward. Instead, we encourage you use this package
inside `Jupyter notebooks`_ .



Contents:

.. toctree::
   :maxdepth: 3

   install
   tutorials
   approach
   input-formats
   _api/epistasis


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Jupyter notebooks: http://jupyter.org/
