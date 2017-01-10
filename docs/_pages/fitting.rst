Fitting
=======

All models in the ``epistasis`` package inherit scikit-learn classes and follow
a scikit-learn interface. If you are already familiar with the

Fit a linear, high-order epistasis model
----------------------------------------

``EpistasisLinearRegression`` is the base class for fitting epistasis in linear genotype-phenotype
maps. It uses an ordinary least squares regression approach to estimate epistatic coefficients
from a list of genotypes-phenotypes pairs. It inherits Scikit-learn's ``LinearRegression``
class and follows the same API. (All attributes and methods are the same.) You can reference
their Docs for more information about the regression aspect of these models.

The ``EpistasisLinearRegression`` class extends scikit-learn's models to fit
epistatic coefficients in genotype-phenotype maps specifically. This means, it creates its own **X** matrix
argument if you don't explicitly pass an ``X`` argument into the ``fit`` method. Further, it appends
a ``GenotypePhenotypeMap`` (see ``gpmap`` package) and ``EpistasisMap`` objects to the model, making
analyzing the data easier and more intuitive.

Example
~~~~~~~

.. code-block:: python

    from epistasis.models import EpistasisLinearRegression

    # Define a genotype-phenotype map
    wildtype="00"
    genotypes = ["00", "01", "10", "00"]
    phenotypes = [0.0,0.5,0.5,1.0]
    order = 2

    # Initialize a model
    model = EpistasisLinearRegression(wildtype, genotypes, phenotypes)

    # Fit the model
    model.fit()

    # Print the epistatic coefs
    print(model.epistasis.values)

Fit a nonlinear epistasis model
-------------------------------

Often, the genotype-phenotype map is nonlinear. That is to say, the genotypes and
phenotypes change on different scales. Genotypes, for example, differ by discrete,
linear changes in sequences (known as mutations). How these changes translate to
phenotype may be less obvious. Sometimes, the effects of mutations simply add together.
Sometimes, the effects multiply. Other times, they change in some other nonlinear
way that is not known a priori. To estimate epistatic coefficients, the genotype-phenotype
map must be on a linear scale.

``EpistasisNonlinearRegression`` class enables you to estimate the scale of any
arbitrary genotype-phenotype map. Simply define the nonlinear relationship you'd expect,
or use some reasonable function that evaluates the shape (i.e. a Box-Cox transform).
The ``EpistasisNonlinearRegression`` will regress this relationship using a nonlinear
least squares regression (using scipy's ``curve_fit`` function), effectivley minimizing
epistasis that might arise from the nonlinear relationship. It can, then, compute
the linearized phenotypes.

Example
~~~~~~~

.. code-block:: python

    from epistasis.models import NonlinearEpistasisRegression

    # Define the nonlinear relationship and it's inverse.
    def boxcox(x, lmbda):
        return (x**lmbda - 1) / lmbda

    def reverse_boxcox(y, lmbda):
        return (lmbda*y + 1) ** (1/lmbda)

    # Initialize the model
    model = NonlinearEpistasisRegression.from_json("data.json",
        order=1,
        function=boxcox,
        reverse=reverse_boxcox
    )

    # Fit the model.
    model.fit(lmbda=3)

The ``epistasis`` package also ships with widgets (via ``ipywidgets``) that aid
in guessing initial values for the nonlinear fit. This is incredibly useful if you
are finding that the nonlinear model isn't converging, or is converging to a local
minimum in the parameter space.

.. code-block:: python

    model.fit(lmbda=3, use_widgets=True)

Fit a multiplicative, high-order epistasis model
------------------------------------------------

Multiplicative epistasis (the effects of mutations multiply together) is a
common nonlinear, phenotypic scale. The following example shows how to estimate
epistasis from a multiplicative scale, using a simple trick of exponentials and
logarithms.

.. math::

    \begin{eqnarray}
    p & = & \beta_1 \beta_2 \beta_{1,2} \\
    p & = & e^{ln(\beta_1 \beta_2 \beta_{1,2})} \\
    p & = & e^{(ln \beta_1 + ln \beta_2 + ln \beta_{1,2})}\\
    p & = & e^{(\alpha_1 + \alpha_2 + \alpha_{1,2})}\\
    \end{eqnarray}
    \text{where } e^{\alpha} = \beta

Example
~~~~~~~

.. code-block:: python

    import numpy as np
    from epistasis.models import NonlinearEpistasisRegression

    # Define the nonlinear relationship and it's inverse.
    def exp(x):
        return np.exp(x)

    def log(y):
        return np.log(y)

    # Initialize the model
    model = NonlinearEpistasisRegression.from_json("data.json",
        order=1,
        function=exp,
        reverse=log
    )

    # Fit
    model.fit()

    # print multiplicative coefficients
    alphas = model.epistasis.values
    betas = np.exp(alphas)


Estimating uncertainty in parameters via bootstrap
--------------------------------------------------
All models have a ``bootstrap_fit`` method to estimate the uncertainty in the
epistatic parameters. This is necessary for interpreting the statistical significance
of the epistatic coefficients and useful for predicting unseen phenotypes.

Fitting a high-order, nonlinear epistasis model
-----------------------------------------------
