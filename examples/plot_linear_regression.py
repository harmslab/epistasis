"""
High-order epistasis model
==========================

``EpistasisLinearRegression`` is the base class for fitting epistasis in linear genotype-phenotype
maps. It uses an ordinary least squares regression approach to estimate epistatic coefficients
from a list of genotypes-phenotypes pairs.

It inherits Scikit-learn's ``LinearRegression``
class and follows the same API. (All attributes and methods are the same.) You can reference
their Docs for more information about the regression aspect of these models.

The ``EpistasisLinearRegression`` class extends scikit-learn's models to fit
epistatic coefficients in genotype-phenotype maps specifically. This means, it creates its own **X** matrix
argument if you don't explicitly pass an ``X`` argument into the ``fit`` method.
"""
# Imports
import matplotlib.pyplot as plt

from gpmap import GenotypePhenotypeMap
from epistasis.models import EpistasisLinearRegression
from epistasis.pyplot import plot_coefs


# The data
wildtype = "000"
genotypes = ['000', '001', '010', '011', '100', '101', '110', '111']
phenotypes = [ 0.366, -0.593,  1.595, -0.753,  0.38 ,  1.296,  1.025, -0.519]
gpm = GenotypePhenotypeMap(wildtype, genotypes, phenotypes)

# Initialize a model
model = EpistasisLinearRegression(order=3)
model.add_gpm(gpm)

# Fit the model
model.fit()

fig, ax = plot_coefs(model, figsize=(2,3))
plt.show()
