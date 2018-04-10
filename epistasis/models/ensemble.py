import numpy as np
import pandas as pd
import lmfit

from sklearn.base import BaseEstimator, RegressorMixin


from epistasis.stats import pearson
from ..mapping import EpistasisMap, mutations_to_sites
from .base import BaseModel
from epistasis.matrix import get_model_matrix
from .utils import arghandler


class State(EpistasisMap):
    """A state in an EpistasisEnsembleModel.

    This represents/models an unknown excited state that is perturbed by
    mutations in the genotype-phenotype map and contributes to the overall
    phenotype.

    Parameters
    ----------
    name : string
        name of the state. Should have the form 'state_X' where X is a letter.

    sites : list of lists
        each element in list is a list of genotype site indexes. These sites
        are additive/epistatic coefficients contributing to this state.

    Attributes
    ----------
    See EpistasisMap for more details.

    keys : list
        list of lmfit.Parameter keys.
    """
    def __init__(self, name, sites, *args, **kwargs):
        # Call super init.
        super(State, self).__init__(sites=sites, *args, **kwargs)

        # Set name.
        self.name = name

        # Construct parameters object
        self.parameters = lmfit.Parameters()

        # fill parameters .
        for key in self.keys:
            self.parameters.add(key, value=0)

    @property
    def keys(self):
        """Coefficient Parameter keys."""
        keys = []
        for sites in self.sites:
            key = "".join([str(ch) for ch in sites])
            name = "{}_{}".format(self.name, key)
            keys.append(name)
        return keys


class EpistasisEnsembleRegression(BaseModel):
    """Ensemble epistasis model. It models variation in a genotype-phenotype map
    as a statistical ensemble of nstates contributing to each genotype's
    phenotype.

    Attributes
    ----------
    See BaseModel for more details.

    parameters : lmfit.Parameters
        Parameters resulting from fit.
    """

    _ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    def __init__(self, order=1, nstates=2):
        self.nstates = nstates
        self.model_type = 'local'
        self.order = order
        self.states = {}
        self.Xbuilt = {}
        self.parameters = lmfit.Parameters()

    def add_gpm(self, gpm):
        """Add genotype-phenotype map to model object."""
        super(EpistasisEnsembleRegression, self).add_gpm(gpm)

        # Add states to model.
        for i in range(self.nstates):
            # State name
            name = "state_{}".format(self._ALPHABET[i])

            # Add state.
            self.add_state(name)

        return self

    def add_state(self, name):
        """ Add a state to the model."""
        sites = self.Xcolumns

        # Create state.
        state = State(name, sites, model_type=self.model_type)

        # Store state.
        self.states[name] = state

        # Set as attribute.
        setattr(self, name, state)

        return self

    @property
    def num_of_params(self):
        """Return number of parameters in model."""
        n = 0
        n += len(self.parameters)
        return n

    @property
    def parameters(self):
        """All parameters in the value."""
        parameters = lmfit.Parameters()

        # Get parameter data.
        for state in self.states.values():
            parameters.add_many(*state.parameters.values())

        return parameters

    @parameters.setter
    def parameters(self, parameters):
        """Set parameters for all states."""

        # Add state parameters.
        for state in self.states.values():
            keys = state.keys
            parameters_ = lmfit.Parameters()
            values = []
            for key in state.keys:
                p = parameters[key]
                parameters_.add_many(p)
                values.append(p.value)

            parameters_.add_many(*[parameters[key] for key in state.keys])
            state.parameters = parameters_
            state.values = values

    def functional_form(self, thetas, X=None):
        """Ensemble function calculating phenotypes using a Boltzmann weighted
        ensemble of states, where each state is constructed from linear
        epistasis models.

        Parameters
        ----------
        thetas : array
            array of parameters values ordered alphabetically.

        X : 2d-array (default None)
            X matrix to use for linear portion of models. If None, uses the
            matrix stored under the 'fit' key.
        """
        length = self.states['state_A'].n
        nstates = len(self.states)

        if X is None:
            X = self.Xbuilt['fit']

        # Calculate a partition function
        Z = 0
        for state_i in range(nstates):
            # Get parameter indexes
            idx_start = state_i * length
            idx_stop = state_i * length + length
            dDG = thetas[idx_start:idx_stop]

            # Get state.
            state = self.states['state_{}'.format(self._ALPHABET[state_i])]

            # Additive model.
            additive = X @ dDG

            # add to ensemble
            Z += np.exp(-additive)

        # Ensemble model.
        y = np.log(Z)
        return y

    @arghandler
    def fit(self, X=None, y=None, **kwargs):
        """Fit ensemble model to data.

        Parameters
        ----------
        X : 2-d array
            independent data; samples.
        y : array
            dependent data; observations.
        """
        # X matrix.
        self.add_X(X=X, key='fit')

        # Storing failed residuals
        last_residual_set = None

        # Residual function to minimize.
        def residual(params, func, y=None):
            # Fit model
            parvals = list(params.values())
            ymodel = func(parvals)

            # Store items in case of error.
            nonlocal last_residual_set
            last_residual_set = (params, ymodel)

            return y - ymodel

        y = self.gpm.phenotypes

        # Minimize the above residual function.
        self.results = lmfit.minimize(
            residual, self.parameters,
            args=[self.functional_form],
            kws={'y': y})

        # Set parameters fitted by model.
        self.parameters = self.results.params

        return self

    def fit_transform(self, X=None, y=None, **kwargs):
        """Same as calling fit in ensemble model.
        """
        self.fit(X=X, y=y, **kwargs)

    @arghandler
    def predict(self, X=None):
        """Predict phenotypes using fitted model.
        """
        return self.functional_form(list(self.parameters.values()), X=X)

    def predict_transform(self, X=None, y=None):
        """Same as calling predict."""
        return self.predict(X=X)

    @arghandler
    def score(self, X=None, y=None, **kwargs):
        """Calculate the pearson coefficient between the models predictions and
        a given y array.
        """
        return pearson(self.gpm.phenotypes, self.predict(X=X))**2
