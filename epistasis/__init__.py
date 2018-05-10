"""\
A Python API for modeling statistical, high-order epistasis in genotype-phenotype maps.
This library provides methods for:

    1. Decomposing genotype-phenotype maps into high-order epistatic interactions
    2. Finding nonlinear scales in the genotype-phenotype map
    3. Calculating the contributions of different epistatic orders
    4. Estimating the uncertainty of epistatic coefficients amd
    5. Interpreting the evolutionary importance of high-order interactions.

For more information about the epistasis models in this library, see our Genetics paper:

    `Sailer, Z. R., & Harms, M. J. (2017). "Detecting High-Order Epistasis in Nonlinear Genotype-Phenotype Maps." Genetics, 205(3), 1079-1088.`_

.. _`Sailer, Z. R., & Harms, M. J. (2017). "Detecting High-Order Epistasis in Nonlinear Genotype-Phenotype Maps." Genetics, 205(3), 1079-1088.`: http://www.genetics.org/content/205/3/1079

Currently, this package works only as an API and there is no command-line
interface. Instead, we encourage you use this package inside `Jupyter notebooks`_ .
"""
from .__version__ import __version__

# from . import models
# from . import simulate
# from . import sampling
# from . import pyplot
