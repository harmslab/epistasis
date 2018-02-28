import numpy as np
import pandas as pd
import lmfit

from epistasis.stats import pearson
from ..mapping import EpistasisMap, mutations_to_sites
from epistasis.model_matrix_ext import get_model_matrix


class Microstate(EpistasisMap):
    """"""
    def __init__(self, name, gpm, *args, **kwargs):
        self.name = name
        self.gpm = gpm
        self.Xbuilt = {}
        self.order = 1
        sites = mutations_to_sites(self.order, self.gpm.mutations)
        super(Microstate, self).__init__(sites=sites, *args, **kwargs)

    @property
    def lmfit_keys(self):
        keys = []
        for sites in self.sites:
            key = "".join([str(ch) for ch in sites])
            name = "{}_{}".format(self.name, key)
            keys.append(name)
        return keys

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

            columns = self.sites

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


class EpistasisEnsembleModel(object):
    """
    """

    _ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    def __init__(self, gpm, nstates):
        self.gpm = gpm
        self.nstates = nstates
        self.model_type = 'local'
        #self.mutation_count = np.array([g.count('1') for g in self.gpm.binary], dtype=int)

        self.states = {}
        self.states_keys = ["state_{}".format(a) for a in self._ALPHABET[:self.nstates]]
        for i in range(self.nstates):
            name = "state_{}".format(self._ALPHABET[i])
            self.states[name] = Microstate(name, self.gpm, model_type=self.model_type)

        # Give each microstate an X.
        self.parameters = lmfit.Parameters()
        for key in self.states_keys:
            microstate = self.states[key]
            microstate.add_X(X='obs', key='fit')
            for name in microstate.lmfit_keys:
                self.parameters.add(name, value=0, max=50, min=-50)

    def _ensemble_model(self, thetas):
        """Ensemble model.
        """
        length = self.states['state_A'].n
        nstates = len(self.states)

        # Calculate a partition function
        Z = 0
        for state_i in range(nstates):
            # Get parameter indexes
            idx_start = state_i * length
            idx_stop = state_i * length + length
            dDG = thetas[idx_start:idx_stop]


            # Get state.
            state = self.states['state_{}'.format(self._ALPHABET[state_i])]
            X = state.Xbuilt['fit']

            # Additive model.
            additive = X @ dDG

            # add to ensemble
            Z += np.exp(-additive)

        # Ensemble model.
        y = np.log(Z)
        return y

    def fit(self):
        """Fit ensemble model.
        """
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

        self.parameters = self.results.params

        for 

        return self

    def predict(self):
        """Predict model.
        """
        return self._ensemble_model(list(self.parameters.values()))

    def score(self):
        """Score fit.
        """
        return pearson(self.gpm.phenotypes, self.predict())**2
