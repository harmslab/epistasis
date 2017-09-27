The Model Matrix
================

Defining the model matrix (the `X` argument in epistasis models) is the most
important piece of any given model. This matrix maps genotypes/phenotypes to 
epistatic coefficients. 

Constructing this matrix can be a complicated, so this package tries to provide
plenty of ways to simplify this. For example, most methods allow you to pass a
key to the `X` argument and the model knows that to do from there. 

These keys include:

- ``"obs"`` : 
- ``"complete"`` : 
- ``"fit"`` : 
- ``"predict"`` : 
