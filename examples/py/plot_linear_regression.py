"""
Epistasis LinearRegression
==========================

An example of linear, high-order epistasis model.
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
