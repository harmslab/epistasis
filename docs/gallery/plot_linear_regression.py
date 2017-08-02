"""
Epistasis LinearRegression
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
# Import the epistasis class
import matplotlib.pyplot as plt
from epistasis.models import EpistasisLinearRegression
import epistasis.plot

# The data

wildtype = "000"
genotypes = ['000', '001', '010', '011', '100', '101', '110', '111']
phenotypes = [ 0.366, -0.593,  1.595, -0.753,  0.38 ,  1.296,  1.025, -0.519]
mutations = {0:["0","1"],1:["0","1"],2:["0","1"]}

# Initialize a model
model = EpistasisLinearRegression(order=3)

# Add the data to the model
model.add_data(wildtype, genotypes, phenotypes, mutations=mutations)

# Fit the model
model.fit()

# Access the epistatic coefficients
sites = model.epistasis.sites
vals = model.epistasis.values

fig, ax = epistasis.plot.coefs(vals, sites, figsize=(2,3))
plt.show()
