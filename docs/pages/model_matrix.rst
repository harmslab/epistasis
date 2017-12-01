The Model Matrix
================

A critical piece of the epistasis models in this API is the X matrix. 
This matrix maps genotypes to epistatic coefficients. You can read about this matrix 
in this paper_. 

There are two popular X matrices that exist in the epistasis literature, the 
``global`` (or background-averaged) and ``local`` (or biochemical) model. 
All epistasis models in this API takes a ``model_type`` keyword argument 
that tells the model which matrix to use. Read the paper mentioned
above for more information on which model to use.

Constructing these matrices for your dataset is no easy task, 
so the ``epistasis`` tries to make this simple. 
Rather than you constructing the matrix yourself, it offers some handy 
keyword arguments that build X matrices that you might be interested in using.

These keys include:

1. obs_
2. missing_
3. complete_ 
4. fit_ 
5. predict_

See the `Matrix kwargs`_ section below for details about each keyword.

Matrix kwargs
-------------
.. _`Matrix kwargs`:






.. References in this document

.. _paper: http://www.genetics.org/content/205/3/1079 
