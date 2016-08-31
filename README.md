# High Order Epistasis Models/Regressions for Genotype-Phenotype Maps

A python API for modeling statistical, high-order epistasis in large genotype-phenotype maps. All models follow a `scikit-learn` interface, making it easy to integrate `epistasis` models with other pipelines and software. It includes a plotting module built on matplotlib for visualizing high-order interactions and interactive widgets to simplify complex nonlinear fits.

This package includes APIs for both linear and nonlinear epistasis models, described in this [paper](), separating epistasis that arises from global trends in phenotypes from epistasis that arises from specific interactions between mutations. Nonlinear regressions

## Basic examples

A simple example
```python
# Import epistasis model
from epistasis.models import LinearEpistasisModel
# Read data from file and estimate epistasis
model = LinearEpistasisModel.from_json("dataset.json")
model.fit()
# Estimate the uncertainty in epistatic coefficients
model.fit_error()
```
More demos are available as [binder notebooks]().

## Installation

To install, clone these repo and run:

```python setup.py install```

or, if you'd like to soft install for development:

```python setup.py develop```

This package is still really hacked together. I plan to include examples and clean up some of the plotting/network managing very soon.

Works in Python 2.7+ and Python 3+

## API reference

API documentation can be viewed [here](http://epistasis.readthedocs.io/).

## Dependencies

* [Seqspace](https://github.com/harmslab/seqspace): Module for constructing powerful genotype-phenotype map python data-structures.
* [Scikit-learn](http://scikit-learn.org/stable/): Simple to use machine-learning algorithms
* [Numpy](http://www.numpy.org/): Python's array manipulation packaged
* [Scipy](http://www.scipy.org/): Efficient scientific array manipulations and fitting.

### Optional dependencies

* [matplotlib](): Python plotting API.
* [ipython](): interactive python kernel.
* [jupyter notebook](): interactive notebook application for running python kernels interactively.   
* [ipywidgets](): interactive widgets in python.

## Citations
If you use this API for research, please cite this [paper]().
