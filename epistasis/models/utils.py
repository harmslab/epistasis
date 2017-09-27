import numpy as np
import pandas as pd
from functools import wraps
from epistasis.model_matrix_ext import get_model_matrix
from epistasis.mapping import EpistasisMap, mutations_to_sites 

import warnings
# Suppresse the future warnings given by X_fitter function.
#warnings.simplefilter(action='ignore', category=FutureWarning)

class XMatrixException(Exception):
    """Exception Subclass for X matrix errors."""

class FittingError(Exception):
    """Exception Subclass for X matrix errors."""

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
    """Wraps a 'scikit-learn'-like predict method with a function that creates
    an X matrix for regression. 
    
    X must be:
        
        - 'obs' : Uses `gpm.binary.genotypes` to construct X. If genotypes are missing
            they will not be included in fit. At the end of fitting, an epistasis map attribute
            is attached to the model class.
        - 'complete' : Uses `gpm.binary.complete_genotypes` to construct X. All genotypes
            missing from the data are included. Warning, will break in most fitting methods.
            At the end of fitting, an epistasis map attribute is attached to the model class.
        - numpy.ndarray : 2d array. Columns are epistatic coefficients, rows are genotypes.
        - pandas.DataFrame : Dataframe with columns labelled as epistatic coefficients, and
            rows labelled by genotypes.
    
    """
    @wraps(method)
    def inner(self, X='complete', *args, **kwargs):
        
        ######## Handle X
        try:
            x = self.Xbuilt[X]
            self.Xbuilt["predict"] = x
            # Run fit.
            return method(self, X=x, *args, **kwargs)

        except (KeyError, TypeError):

            if type(X) is str and X in ['obs', 'complete']:
                
                if hasattr(self, "gpm") == False:
                    raise XMatrixException("To build 'obs' or 'complete' X matrix, "
                                           "a GenotypePhenotypeMap must be attached.")
                
                # Construct an X for this model.
                x = self.add_X(X=X)
                
                # # Build epistasis interactions as columns in X matrix.
                # columns = mutations_to_sites(self.order, self.gpm.mutations)
                # 
                # # Use desired set of genotypes for rows in X matrix.        
                # if X == "obs":
                #     index = self.gpm.binary.genotypes
                # else:
                #     index = self.gpm.binary.complete_genotypes
                # 
                # # Build numpy array
                # x = get_model_matrix(index, columns, model_type=self.model_type)
                # 
                
                # Store Xmatrix.
                self.Xbuilt[X] = x
                self.Xbuilt["predict"] = x
                # Run fit.
                prediction = method(self, X=x, *args, **kwargs)

            elif type(X) == np.ndarray or type(X) == pd.DataFrame: 
                                
                # Store Xmatrix.
                self.Xbuilt["predict"] = X
                prediction = method(self, X=X, *args, **kwargs)

            else:
                raise XMatrixException("X must be one of the following: 'obs', 'complete', "
                                       "numpy.ndarray, or pandas.DataFrame.")

        return prediction

    return inner


def X_fitter(method):
    """Wraps a 'scikit-learn'-like fit method with a function that creates
    an X matrix for regression. Also, saves all X matrices in the `Xbuilt` attribute.
    
    X must be:
        
        - 'obs' : Uses `gpm.binary.genotypes` to construct X. If genotypes are missing
            they will not be included in fit. At the end of fitting, an epistasis map attribute
            is attached to the model class.
        - 'complete' : Uses `gpm.binary.complete_genotypes` to construct X. All genotypes
            missing from the data are included. Warning, will break in most fitting methods.
            At the end of fitting, an epistasis map attribute is attached to the model class.
        - numpy.ndarray : 2d array. Columns are epistatic coefficients, rows are genotypes.
        - pandas.DataFrame : Dataframe with columns labelled as epistatic coefficients, and
            rows labelled by genotypes.
            
            
    y must be:
        - 'obs' : Uses `gpm.binary.phenotypes` to construct y. If phenotypes are missing
            they will not be included in fit. 
        - 'complete' : Uses `gpm.binary.complete_genotypes` to construct X. All genotypes
            missing from the data are included. Warning, will break in most fitting methods.
        - 'fit' : a previously defined array/dataframe matrix. Prevents copying for efficiency.
        - numpy.array : 1 array. List of phenotypes. Must match number of rows in X.
        - pandas.DataFrame : Dataframe with columns labelled as epistatic coefficients, and
            rows labelled by genotypes.
    """
    @wraps(method)
    def inner(self, X='obs', y='obs', *args, **kwargs):
        
        ######## Sanity checks on input.
                
        # Make sure X and y strings match
        if type(X) == str and type(y) == str and X != y:
            raise FittingError("Any string passed to X must be the same as any string passed to y. "
                           "For example: X='obs', y='obs'.")
            
        # Else if both are arrays, check that X and y match dimensions.
        elif type(X) != str and type(y) != str and X.shape[0] != y.shape[0]:
            raise FittingError("X dimensions {} and y dimensions {} don't match.".format(X.shape[0], y.shape[0]))
            
        ######## Handle y.
        
        # Check if string.
        if type(y) is str and y in ["obs", "complete"]:            

            y = self.gpm.binary.phenotypes

        # Else, numpy array or dataframe
        elif type(y) != np.array and type(y) != pd.Series:
            
            raise FittingError("y is not valid. Must be one of the following: 'obs', 'complete', "
                           "numpy.array, pandas.Series. Right now, its {}".format(type(y)))    
                
        ######## Handle X
        try:
            x = self.Xbuilt[X]
            # Run fit.
            model = method(self, X=x, y=y, *args, **kwargs)
            self.Xbuilt["fit"] = x
                        
        except (KeyError, TypeError):
                
            if type(X) is str and X in ['obs', 'complete']:
                
                if hasattr(self, "gpm") == False:
                    raise XMatrixException("To build 'obs' or 'complete' X matrix, "
                                           "a GenotypePhenotypeMap must be attached.")
                
                # Enforce that a new EpistasisMap is built.
                self.add_epistasis()

                # Construct an X for this model.
                x = self.add_X(X=X)
                
                # Store Xmatrix.                
                # (if the GenotypePhenotypeMap is complete (no missig genotypes), then
                # the 'obs' X == 'complete' X)
                if len(self.gpm.binary.missing_genotypes) == 0:
                    self.Xbuilt["complete"] = x
                    self.Xbuilt["obs"] = x
                    
                else:
                    self.Xbuilt[X] = x
                
                # Run fit.
                model = method(self, X=x, y=y, *args, **kwargs)
                                
                # Store Xmatrix.
                self.Xbuilt["fit"] = x
                self.epistasis.values = np.reshape(self.coef_,(-1,))
            
            elif type(X) == np.ndarray or type(X) == pd.DataFrame: 
                # Call method with X and y.
                model = method(self, X=X, y=y, *args, **kwargs)
                
                # Store Xmatrix.
                self.Xbuilt["fit"] = X
            
            else:
                raise XMatrixException("X must be one of the following: 'obs', 'complete', "
                                       "numpy.ndarray, or pandas.DataFrame.")

        # Return model
        return model
    
    return inner
