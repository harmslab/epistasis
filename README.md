# High Order Epistasis Models/Regressions for Genotype-Phenotype Maps

[![Join the chat at https://gitter.im/harmslab/epistasis](https://badges.gitter.im/harmslab/epistasis.svg)](https://gitter.im/harmslab/epistasis?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/harmslab/epistasis)
[![Documentation Status](https://readthedocs.org/projects/epistasis/badge/?version=latest)](http://epistasis.readthedocs.io/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.242665.svg)](https://doi.org/10.5281/zenodo.242665)

A python API for modeling statistical, high-order epistasis in genotype-phenotype maps. All models follow a *Scikit-learn* interface, making it easy to integrate `epistasis` models with other pipelines and software. It includes a plotting module built on matplotlib for visualizing high-order interactions and interactive widgets to simplify complex nonlinear fits.

This package includes APIs for both linear and nonlinear epistasis models, described in this [paper](http://biorxiv.org/content/early/2016/12/02/072256), relaxing
the assumption of linearity.  

## Basic examples

A simple example of fitting a data set with a linear epistasis model.  
```python
# Import epistasis model
from epistasis.models import EpistasisLinearRegression

# Read data from file and estimate epistasis
model = EpistasisLinearRegression.from_json("dataset.json", order=3)
model.fit()

# Estimate the uncertainty in epistatic coefficients
model.bootstrap_fit()
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
model.fit_widget(lmbda=(-2,2,.1), lmbda2=(-2,2,.1))
```

More demos are available as [binder notebooks](http://mybinder.org/repo/harmslab/epistasis).

## Installation

To install, clone these repo and run:

```
python setup.py install
```

or, if you'd like to soft install for development:

```
python setup.py develop
```

This package is still really hacked together. I plan to include examples and clean up some of the plotting/network managing very soon.

Works in Python 2.7+ and Python 3+

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

## Citations
If you use this API for research, please cite this [paper](http://biorxiv.org/content/early/2016/12/02/072256).
