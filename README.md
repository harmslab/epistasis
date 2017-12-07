# Epistasis

[![Join the chat at https://gitter.im/harmslab/epistasis](https://badges.gitter.im/harmslab/epistasis.svg)](https://gitter.im/harmslab/epistasis?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Binder](http://mybinder.org/badge.svg)](https://beta.mybinder.org/v2/gh/harmslab/epistasis-notebooks/master)
[![Documentation Status](https://readthedocs.org/projects/epistasis/badge/?version=latest)](http://epistasis.readthedocs.io/?badge=latest)
[![Build Status](https://travis-ci.org/harmslab/epistasis.svg?branch=master)](https://travis-ci.org/harmslab/epistasis)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.242665.svg)](https://doi.org/10.5281/zenodo.242665)

*Python API for estimating statistical, high-order epistasis in genotype-phenotype maps.*

All models follow a *Scikit-learn* interface and thus seamlessly plug in to the PyData ecosystem. For more information about the type of models included in this package,
read our [docs](http://epistasis.readthedocs.io/?badge=latest). You can also read more about the theory behind these models in our [paper](https://doi.org/10.1534/genetics.116.195214).

Finally, if you'd like to test out this package without any installing, try these Jupyter notebooks [here](https://mybinder.org/v2/gh/harmslab/epistasis-notebooks/master) (thank you [Binder](https://mybinder.org/)!).

## Examples

The Epistasis package works best in combinations with GPMap, an API for managing
genotype-phenotype map data. Construct a GenotypePhenotypeMap object and pass it
directly to an epistasis model.

```python
# Import gpmap
from gpmap import GenotypePhenotypeMap

# Import epistasis model
from epistasis.models import EpistasisLinearRegression

# Load genotype-phenotype map data
gpm = GenotypePhenotypeMap.read_json()

# Initialize model and add data,
model = EpistasisLinearRegression(order=3, model_type='global')
model.add_gpm(gpm)

# Fit the model
model.fit()
```

More examples can be found in these [binder notebooks](https://mybinder.org/v2/gh/harmslab/epistasis-notebooks/master).

## Installation

Epistasis works in Python 3+ (we do not guarantee it will work in Python 2.)

To install the most recent release on PyPi:
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

To run the tests on this package, make sure you have `pytest` installed and run from the base directory:

```
pytest
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
