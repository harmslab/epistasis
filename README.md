# Python API for estimating statistical, high-order epistasis

[![Join the chat at https://gitter.im/harmslab/epistasis](https://badges.gitter.im/harmslab/epistasis.svg)](https://gitter.im/harmslab/epistasis?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Binder](http://mybinder.org/badge.svg)](https://beta.mybinder.org/v2/gh/harmslab/epistasis)
[![Documentation Status](https://readthedocs.org/projects/epistasis/badge/?version=latest)](http://epistasis.readthedocs.io/?badge=latest)
[![Build Status](https://travis-ci.org/harmslab/epistasis.svg?branch=master)](https://travis-ci.org/harmslab/epistasis)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.242665.svg)](https://doi.org/10.5281/zenodo.242665)



A python API for estimating statistical, high-order epistasis in linear and nonlinear genotype-phenotype maps. All models follow a *Scikit-learn* interface, making it easy to integrate `epistasis` models with other pipelines and software. It includes a plotting module built on matplotlib for visualizing high-order interactions and interactive widgets to simplify complex nonlinear fits.

This package includes APIs for both linear and nonlinear epistasis models, described in this [paper](https://doi.org/10.1534/genetics.116.195214).

If you'd like to see how we used the epistasis package in our recent Genetics paper (2017), run our Jupyter notebooks [here](http://mybinder.org:/repo/harmslab/notebooks-nonlinear-high-order-epistasis)!

## Basic examples

A simple example of fitting a data set with a linear epistasis model.  
```python
# Import epistasis model
from epistasis.models import EpistasisLinearRegression

# Read data from file and estimate epistasis
model = EpistasisLinearRegression.from_json("dataset.json", order=3)
model.fit()
```

If analyzing a nonlinear genotype-phenotype map, use `NonlinearEpistasisModel`
(nonlinear least squares regression) to estimate nonlinearity in map:
```python
# Import the nonlinear epistasis model
from epistasis.models import NonlinearEpistasisRegression

# Define a nonlinear function to fit the genotype-phenotype map.
def boxcox(x, lmbda, lmbda2):
    """Fit with a box-cox function to estimate nonlinearity."""
    return ((x-lmbda2)**lmbda - 1 )/lmbda

def reverse_boxcox(y, lmbda, lmbda2):
    "inverse of the boxcox function."
    return (lmbda*y + 1) ** (1/lmbda) + lmbda2

# Read data from file and estimate nonlinearity in dataset.
model = EpistasisNonlinearRegression.from_json("dataset.json",
    function=boxbox,
    reverse=reverse_boxcox,
    order=1,
)

# Give initial guesses for parameters to aid in convergence (not required).
model.fit(lmbda=1, lmbda2=1)
```

The nonlinear fit also includes Jupyter Notebook widgets to make nonlinear fitting
easier.
```python
model.fit(lmbda=(-2,2,.1), lmbda2=(-2,2,.1), use_widgets=True)
```

More demos are available as [binder notebooks](http://mybinder.org/repo/harmslab/epistasis).

## Installation

You must have Python 2.7+ or 3+ installed.

To install the most recent release of this package, run:
```
pip install epistasis
```

To install from source, clone this repo and run:
```
pip install -e .
```

## Documentation

Documentation and API reference can be viewed [here](http://epistasis.readthedocs.io/).

## Dependencies

* [gpmap](https://github.com/harmslab/gpmap): Module for constructing powerful genotype-phenotype map python data-structures.
* [Scikit-learn](http://scikit-learn.org/stable/): Simple to use machine-learning algorithms
* [Numpy](http://www.numpy.org/): Python's array manipulation packaged
* [Scipy](http://www.scipy.org/): Efficient scientific array manipulations and fitting.

### Optional dependencies

* [matplotlib](): Python plotting API.
* [ipython](): interactive python kernel.
* [jupyter notebook](): interactive notebook application for running python kernels interactively.   
* [ipywidgets](): interactive widgets in python.

## Development

We welcome pull requests! If you find a bug, we'd love to have you fix it. If
there is a feature you'd like to add, feel free to submit a
pull request with a description of the addition. We also ask that you write the
appropriate unit-tests for the new feature and add documentation to our Sphinx docs.

To run the tests on this package, run nose tests on the command line:

```
nosetests
```

## Citing
If you use this API for research, please cite this [paper](https://doi.org/10.1534/genetics.116.195214).

You can also cite the software directly:

```
@misc{zachary_sailer_2017_252927,
  author       = {Zachary Sailer and Mike Harms},
  title        = {harmslab/epistasis: Genetics paper release},
  month        = jan,
  year         = 2017,
  doi          = {10.5281/zenodo.252927},
  url          = {https://doi.org/10.5281/zenodo.252927}
}
```
