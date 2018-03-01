import numpy as np
import pandas as pd
import lmfit

from epistasis.stats import pearson
from ..mapping import EpistasisMap, mutations_to_sites
from .base import BaseModel
from epistasis.model_matrix_ext import get_model_matrix
from .utils import X_fitter, X_predictor


class State(EpistasisMap):
    """State to model in an ensemble."""
    def __init__(self, name, sites, *args, **kwargs):
        # Call super init.
        super(State, self).__init__(sites=sites, *args, **kwargs)

        # Set name.
        self.name = name

        # Construct parameters object
        self.parameters = lmfit.Parameters()

        # fill parameters .
        for key in self.keys:
            self.parameters.add(key, max=50, min=-50, value=0)

    @property
    def keys(self):
        """State coefficient Parameter keys."""
        keys = []
        for sites in self.sites:
            key = "".join([str(ch) for ch in sites])
            name = "{}_{}".format(self.name, key)
            keys.append(name)
        return keys


class EpistasisEnsembleModel(BaseModel):
    """Ensemble model.

    Attributes
    ----------
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
        super(EpistasisEnsembleModel, self).add_gpm(gpm)

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


    def add_X(self, X="complete", key=None):
        """Add X to Xbuilt

        Keyword arguments for X:

        - 'obs' :
            Uses ``gpm.binary`` to construct X. If genotypes
            are missing they will not be included in fit. At the end of
            fitting, an epistasis map attribute is attached to the model
            class.
        - 'missing' :
            Uses ``gpm.binary`` to construct X.
            All genotypes missing from the data are included. Warning,
            will break in most fitting methods. At the end of fitting,
            an epistasis map attribute is attached to the model class.
        - 'complete' :
            Uses ``gpm.binary`` to construct X.
            All genotypes missing from the data are included. Warning, will
            break in most fitting methods. At the end of fitting, an
            epistasis map attribute is attached to the model class.
        - 'fit' :
            a previously defined array/dataframe matrix. Prevents
            copying for efficiency.


        Parameters
        ----------
        X :
            see above for details.
        key : str
            name for storing the matrix.

        Returns
        -------
        Xbuilt : numpy.ndarray
            newly built 2d array matrix
        """
        if type(X) is str and X in ['obs', 'missing', 'complete', 'fit']:

            if hasattr(self, "gpm") is False:
                raise XMatrixException("To build 'obs', 'missing', or"
                                       "'complete' X matrix, a "
                                       "GenotypePhenotypeMap must be attached")

            # Use desired set of genotypes for rows in X matrix.
            if X == "obs":
                index = self.gpm.binary
            elif X == "missing":
                index = self.gpm.missing_binary
            else:
                index = self.gpm.complete_binary

            columns = self.state_A.sites

            # Build numpy array
            x = get_model_matrix(index, columns, model_type=self.model_type)

            # Set matrix with given key.
            if key is None:
                key = X

            self.Xbuilt[key] = x

        elif type(X) == np.ndarray or type(X) == pd.DataFrame:
            # Set key
            if key is None:
                raise Exception("A key must be given to store.")

            # Store Xmatrix.
            self.Xbuilt[key] = X

        else:
            raise XMatrixException("X must be one of the following: 'obs', "
                                   "'complete', numpy.ndarray, or "
                                   "pandas.DataFrame.")

        Xbuilt = self.Xbuilt[key]
        return Xbuilt

    def _ensemble_model(self, thetas, X=None):
        """Ensemble model.
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

    @X_fitter
    def fit(self, X='obs', y='obs', **kwargs):
        """Fit ensemble model.
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
            args=[self._ensemble_model],
            kws={'y': y})

        # Set parameters fitted by model.
        self.parameters = self.results.params

        return self

    @X_predictor
    def predict(self, X='complete'):
        """Predict model.
        """
        return self._ensemble_model(list(self.parameters.values()), X=X)

    @X_fitter
    def score(self, X='obs', y='obs', **kwargs):
        """Score fit.
        """
        return pearson(self.gpm.phenotypes, self.predict(X=X))**2
