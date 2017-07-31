import numpy as np
from functools import wraps
from epistasis.model_matrix_ext import get_model_matrix
import epistasis.mapping

import warnings
# Suppresse the future warnings given by X_fitter function.
warnings.simplefilter(action='ignore', category=FutureWarning)

def sklearn_to_epistasis():
    """Decorate a scikit learn class with this function and automagically convert it into a
    epistasis sklearn model class.
    """
    def inner(cls):
        base_model = cls.__bases__[-1]
        for attr in base_model.__dict__:
            method = getattr(base_model, attr)
            try:
                setattr(cls, attr, getattr(base_model, attr))
            except AttributeError: pass
        return cls
    return inner

def X_predictor(method):
    """Decorator to automatically generate X for predictor methods in epistasis models."""
    @wraps(method)
    def inner(self, X=None, *args, **kwargs):
        """"""
        # If no X is given, ALWAYS build a new Xpredict. This will not use old Xpredict matrices.
        if X is None and hasattr(self, "Xpredict"):
            X = self.Xpredict
        elif X is None:
            # Construct an X matrix if none is given. Assumes
            genotypes = self.gpm.binary.complete_genotypes
            coefs = self.epistasis.sites
            model_type = self.model_type
            X = get_model_matrix(genotypes, coefs, model_type=model_type)
            self.Xpredict = X

        # Save this matrix for later predictions.
        return method(self, X=X, *args, **kwargs)

    return inner

def X_fitter(method):
    """Wraps a 'scikit-learn'-like fit method with a function that creates
    an X fitter matrix. This requires that a GenotypePhenotypeMap object be attached
    to the model class.
    """
    @wraps(method)
    def inner(self, X=None, y=None, *args, **kwargs):
        # If X and y is not given, a GenotypePhenotypeMap must be attached. If
        # a GenotypePhenotypeMap is not attached, raise an exception.
        if True in (X is None, y is None) and hasattr(self, "gpm") is False:
            raise Exception("If both X and y are not given, a GenotypePhenotypeMap must be attached.")

        # If no Y is given, try to get it from
        if y is None:
            # Pull y from the phenotypes in a GenotypePhenotypeMap
            y = np.array(self.gpm.binary.phenotypes)

        # Prepare X for fit method.
        if X is not None:
            # Must remove any old epistasis map, since model object has no clue
            # what the new model matrix is.
            if hasattr(self, "epistasis"): delattr(self, "epistasis")

            # If X is given, nothing needs to be done.
            model = method(self, X=X, y=y, *args, **kwargs)

        # Check to see if an Xfit matrix already exists.
        elif hasattr(self, "Xfit"):
            X = self.Xfit

            # Reference the model coefficients in the epistasis map.
            model = method(self, X=X, y=y, *args, **kwargs)

        # If no Xfit matrix exists, create one using the genotype phenotype map.
        else:
            # Prepare for construction of new model.
            genotypes = self.gpm.binary.genotypes
            mutations = self.gpm.mutations
            order = self.order
            model_type = self.model_type
            coefs = epistasis.mapping.mutations_to_sites(order, mutations)
            X = get_model_matrix(genotypes, coefs, model_type=model_type)

            # Call fitter method
            model = method(self, X=X, y=y, *args, **kwargs)

            # Assign a nested mapping class to the epistasis attribute if coefs exist.
            #values = np.reshape(self.coef_,(-1,))
            try:
                values = np.reshape(self.coef_,(-1,))
                self.epistasis = epistasis.mapping.EpistasisMap(coefs, order=order, model_type=model_type)
                self.epistasis.values = values
            except AttributeError:
                pass

        # Store the X matrix and return the fit method output.
        self.Xfit = X
        return model

    return inner
