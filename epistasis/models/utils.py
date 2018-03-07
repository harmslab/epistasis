import numpy as np
import pandas as pd
from functools import wraps
from epistasis.model_matrix_ext import get_model_matrix
from epistasis.mapping import EpistasisMap, mutations_to_sites

from gpmap.utils import genotypes_to_binary

import warnings
# Suppresse the future warnings given by X_fitter function.
# warnings.simplefilter(action='ignore', category=FutureWarning)


class XMatrixException(Exception):
    """Exception Subclass for X matrix errors."""


class FittingError(Exception):
    """Exception Subclass for X matrix errors."""


def sklearn_to_epistasis():
    """Decorate a scikit learn class with this function and automagically
    convert it into a epistasis sklearn model class.
    """
    def inner(cls):
        base_model = cls.__bases__[-1]
        for attr in base_model.__dict__:
            method = getattr(base_model, attr)
            try:
                setattr(cls, attr, getattr(base_model, attr))
            except AttributeError:
                pass
        return cls
    return inner


def X_predictor(method):
    """Wraps a 'scikit-learn'-like predict method with a function that creates
    an X matrix for regression.

    X must be:

    - 'obs' :
        Uses `gpm.binary` to construct X. If genotypes
        are missing they will not be included in fit. At the end of
        fitting, an epistasis map attribute is attached to the model class.
    - 'fit':
        Predict using matrix from last fit call.
    - numpy.ndarray :
        2d array. Columns are epistatic coefficients, rows
        are genotypes.
    """
    @wraps(method)
    def inner(self, X='obs', *args, **kwargs):

        # ------- Process the X argument --------
        # First, check if X is stored in model.
        try:
            # Try to get X from Xbuilt.
            x = self.Xbuilt[X]

            # If stored, set the predict key to this matrix.
            self.Xbuilt["predict"] = x

            # Run fit.
            return method(self, X=x, *args, **kwargs)

        # If X is not in Xbuilt, create X.
        except (KeyError, TypeError):

            # If X is a string, it better be 'obs'
            if isinstance(X, str):

                if hasattr(self, "gpm") is False:
                    raise XMatrixException("To build 'obs' or 'complete' X "
                                           "matrix, a GenotypePhenotypeMap "
                                           "must be attached.")

                # Construct an X for this model.
                x = self.add_X(X=X)

                # Store Xmatrix.
                self.Xbuilt[X] = x
                self.Xbuilt["predict"] = x
                # Run fit.
                prediction = method(self, X=x, *args, **kwargs)

            # else if a numpy array, prepare it.
            elif isinstance(X, np.ndarray) or isinstance(X, list):

                # If first element in X is string, it must be a genotype.
                if isinstance(X[0], str):
                    # X must be genotypes
                    genotypes = X

                    # Genotypes to binary
                    binary = genotypes_to_binary(
                        self.gpm.wildtype,
                        genotypes,
                        self.gpm.mutations
                    )

                    # Build list of sites from genotypes.
                    sites = mutations_to_sites(self.order, self.gpm.mutations)

                    # X matrix
                    X = get_model_matrix(binary, sites,
                        model_type=self.model_type
                    )

                # Add X to Xbuilt
                self.Xbuilt['predict'] = X

                # Run prediction method
                prediction = method(self, X=X, *args, **kwargs)

            # Else, raise exception.
            else:
                raise XMatrixException("Bad input for X. Check X.")

        return prediction

    return inner


def X_fitter(method):
    """Wraps a 'scikit-learn'-like fit method with a function that creates
    an X matrix for regression. Also, saves all X matrices in the `Xbuilt`
    attribute.

    X must be:

    - 'obs' :
        Uses `gpm.binary` to construct X. If genotypes are
        missing they will not be included in fit. At the end of fitting, an
        epistasis map attribute is attached to the model class.
    - numpy.ndarray :
        2d array. Columns are epistatic coefficients, rows
        are genotypes.

    y must be:

    - 'obs' :
        Uses `gpm.binary` to construct y. If phenotypes
        are missing they will not be included in fit.
    - 'fit' :
        a previously defined array/dataframe matrix. Prevents copying
        for efficiency.
    - numpy.array :
        1 array. List of phenotypes. Must match number of rows
        in X.
    """
    @wraps(method)
    def inner(self, X='obs', y='obs', sample_weight=None, *args, **kwargs):

        # Sanity checks on input.

        # Make sure X and y strings match
        if type(X) == str and type(y) == str and X != y:
            raise FittingError("Any string passed to X must be the same as any"
                               "string passed to y. For example: X='obs', "
                               "y='obs'.")

        # Else if both are arrays, check that X and y match dimensions.
        elif type(X) != str and type(y) != str and X.shape[0] != y.shape[0]:
            raise FittingError("X dimensions {} and y dimensions {} don't"
                               " match.".format(X.shape[0], y.shape[0]))

        # ------- Process the y argument --------

        # Check if string.
        if type(y) is str and y in ["obs", "fit"]:

            y = self.gpm.phenotypes

        # Else, numpy array or dataframe
        elif type(y) != np.ndarray and type(y) != pd.Series:

            raise FittingError("y is not valid. Must be one of the following: "
                               "'obs', numpy.array, pandas.Series."
                               " Right now, its {}".format(type(y)))

        # Handle sample weights
        if sample_weight == 'relative':
            sample_weight = 1 / abs(y**2)

        # ------- Process the X argument --------
        # First, check if X is stored in model.
        try:
            # Does X exist in Xbuilt?
            x = self.Xbuilt[X]

            # Yes? then run fit.
            model = method(
                self, X=x, y=y, sample_weight=sample_weight, *args, **kwargs)

            # Store X at fit key.
            self.Xbuilt["fit"] = x

        # If not in Xbuilt, process X.
        except (KeyError, TypeError):

            if hasattr(self, "gpm") is False:
                raise XMatrixException(
                    "To build 'obs' or 'complete' X "
                    "matrix, a GenotypePhenotypeMap "
                    "must be attached.")

            # Turn X into matrix.
            if isinstance(X, str):
                # Get key
                x = X
                X = self.add_X(X=x)

                # Store Xmatrix.
                self.Xbuilt[x] = X

            elif isinstance(X, np.ndarray):

                # If first argument is a string, must be a genotype.
                if isinstance(X[0], str):
                    # X must be genotypes
                    genotypes = X

                    # Genotypes to binary
                    binary = genotypes_to_binary(
                        self.gpm.wildtype,
                        genotypes,
                        self.gpm.mutations
                    )

                    # Build list of sites from genotypes.
                    sites = mutations_to_sites(self.order, self.gpm.mutations)

                    # X matrix
                    X = get_model_matrix(binary, sites, model_type=self.model_type)

            else:
                raise XMatrixException('X is an invalid type. Check your X input.')

            # Run fit method with X.
            model = method(self, X=X, y=y,
                           sample_weight=sample_weight,
                           *args, **kwargs)

            self.Xbuilt["fit"] = X

        # Return model
        return model

    return inner


def epistasis_fitter(fit_method):
    """Connect an epistasis object to the model.
    """
    @wraps(fit_method)
    def inner(self, X='obs', *args, **kwargs):
        if type(X) is np.ndarray or type(X) is pd.DataFrame:
            model = fit_method(self, X=X, *args, **kwargs)

        elif X not in self.Xbuilt:
            # Map those columns to epistastalis dataframe.
            self.epistasis = EpistasisMap(
                sites=self.Xcolumns,
                order=self.order,
                model_type=self.model_type)

            # Execute fitting method
            model = fit_method(self, X=X, *args, **kwargs)

            # Link coefs to epistasis values.
            self.epistasis.values = np.reshape(self.coef_, (-1,))

        else:
            model = fit_method(self, X=X, *args, **kwargs)

        return model
    return inner
