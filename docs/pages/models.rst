Epistasis models
================

This page provides short descriptions of each epistasis model available in this library.

Table of contents
-----------------

* EpistasisLinearRegression
* EpistasisNonLinearRegression
* EpistasisPowerTransform
* EpistasisMixedRegression


EpistasisLinearRegression
-------------------------

Ordinary least-squares regression for estimating high-order, epistatic interactions in a genotype-phenotype map.

EpistasisNonLinearRegression
----------------------------

Use nonlinear least-squares regression to estimate epistatic coefficients and nonlinear scale in a nonlinear genotype-phenotype map.

  This models has three steps:
      1. Fit an additive, linear regression to approximate the average effect of individual mutations.
      2. Fit the nonlinear function to the observed phenotypes vs. the additive phenotypes estimated in step 1.
      3. Transform the phenotypes to this linear scale and fit leftover variation with high-order epistasis model.

EpistasisPowerTransform
-----------------------

Use power-transform function, via nonlinear least-squares regression, to estimate epistatic coefficients and the nonlinear scale in a nonlinear genotype-phenotype map.

Like the nonlinear model, this model has three steps:
    1. Fit an additive, linear regression to approximate the average effect of
        individual mutations.
    2. Fit the nonlinear function to the observed phenotypes vs. the additive
        phenotypes estimated in step 1.
    3. Transform the phenotypes to this linear scale and fit leftover variation
        with high-order epistasis model.

Methods are described in the following publication:
    
    Sailer, Z. R. & Harms, M. J. 'Detecting High-Order Epistasis in Nonlinear
    Genotype-Phenotype Maps'. Genetics 205, 1079-1088 (2017).


EpistasisMixedRegression
------------------------

A high-order epistasis regression that first classifies genotypes as viable/nonviable (given some threshold) and then estimates epistatic coefficients in viable phenotypes.
