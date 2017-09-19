.. epistasis documentation master file, created by
   sphinx-quickstart on Thu Jul  7 15:47:18 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
============

A Python API for modeling statistical, high-order epistasis in genotype-phenotype maps.
This library provides methods for:

    1. Decomposing genotype-phenotype maps into high-order epistatic interactions
    2. Finding nonlinear scales in the genotype-phenotype map
    3. Calculating the contributions of different epistatic orders
    4. Estimating the uncertainty of epistatic coefficients amd
    5. Interpreting the evolutionary importance of high-order interactions.

For more information about the epistasis models in this library, see our Genetics paper:

    `Sailer, Z. R., & Harms, M. J. (2017). "Detecting High-Order Epistasis in Nonlinear Genotype-Phenotype Maps." Genetics, 205(3), 1079-1088.`_

.. _`Sailer, Z. R., & Harms, M. J. (2017). "Detecting High-Order Epistasis in Nonlinear Genotype-Phenotype Maps." Genetics, 205(3), 1079-1088.`: http://www.genetics.org/content/205/3/1079

Currently, this package works only as an API and there is no command-line
interface. Instead, we encourage you use this package inside `Jupyter notebooks`_ .

Table of Contents
=================

.. toctree::
   :maxdepth: 2

   pages/install
   pages/models
   pages/fitting
   pages/estimating-uncertainty
   pages/simulate
   pages/plot
   pages/io
   gallery/index.rst
   api/main.rst

Example Gallery
===============

The following gallery contains various examples from the package.


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="``EpistasisLinearRegression`` is the base class for fitting epistasis in linear genotype-phenot...">

.. only:: html

    .. figure:: /gallery/images/thumb/sphx_glr_plot_linear_regression_thumb.png

        :ref:`sphx_glr_gallery_plot_linear_regression.py`

.. raw:: html

    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Use a linear, logistic regression model to estimate the positive/negative effects of mutations....">

.. only:: html

    .. figure:: /gallery/images/thumb/sphx_glr_plot_logistic_regression_thumb.png

        :ref:`sphx_glr_gallery_plot_logistic_regression.py`

.. raw:: html

    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Often, the genotype-phenotype map is nonlinear. That is to say, the genotypes and phenotypes ch...">

.. only:: html

    .. figure:: /gallery/images/thumb/sphx_glr_plot_nonlinear_simulation_thumb.png

        :ref:`sphx_glr_gallery_plot_nonlinear_simulation.py`

.. raw:: html

    </div>


   /gallery/plot_nonlinear_simulation
.. raw:: html

    <div style='clear:both'></div>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Jupyter notebooks: http://jupyter.org/
