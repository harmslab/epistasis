"""
Fitting nonlinear genotype-phenotype maps
=========================================

Use a linear, logistic regression model to estimate the positive/negative effects
of mutations.
"""

# Imports
import matplotlib.pyplot as plt

from gpmap.simulate import MountFujiSimulation

from epistasis.models import EpistasisPowerTransform
from epistasis.pyplot import plot_power_transform

# The data
gpm = MountFujiSimulation.from_length(4, field_strength=-1, roughness=(-1,1))

# Initialize a model
model = EpistasisPowerTransform(lmbda=1, A=0, B=0)
model.add_gpm(gpm)

# Fit the model
model.fit()

fig, ax = plt.subplots(figsize=(2.5,2.5))
ax = plot_power_transform(model)
plt.show()
