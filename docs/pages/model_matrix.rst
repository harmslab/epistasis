The Model Matrix
================

The most important piece of an ``epistasis.model`` is the X matrix. This matrix
maps genotypes to epistatic coefficients. You can read about this matrix 
in this paper_. 

There are two popular X matrices that exist in the epistasis literature, the 
``global`` and ``local`` model. Each epistasis model takes a ``model_type`` 
keyword argument that tells the model which matrix to use. Read the paper mentioned
above for more information on which model to use.

Constructing these models is no easy task, so the ``epistasis`` tries to make this
simple. Rather than you constructing the matrix yourself, it offers some handy 
keyword arguments for building common matrices. 

These keys include:

1. obs_
2. complete_ 
3. fit_ 
4. predict_

See the `Matrix kwargs`_ section below for details about each keyword.

Matrix kwargs
-------------
.. _`Matrix kwargs`:

Th





.. References in this document

.. _paper: http://www.genetics.org/content/205/3/1079 
