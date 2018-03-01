import numpy as np
import pandas as pd
from sklearn.preprocessing import binarize
from sklearn.base import BaseEstimator, RegressorMixin

import epistasis.mapping
from epistasis.model_matrix_ext import get_model_matrix
from epistasis.stats import pearson

from .base import BaseModel
from .power import EpistasisPowerTransform
from .classifiers import EpistasisLogisticRegression
from .utils import FittingError, XMatrixException


class EpistasisMixedRegression(BaseModel, BaseEstimator):
    """An object that links a phenotype classifier model and an epistasis model
    seemlessly to allow classification of dead/alive phenotypes first, then
    estimation of epistatic coefficients between the alive phenotypes.

    Parameters
    ----------
    Classifier : epistasis.models.classifiers object
        Instance of an epistasis Cclassifier.
    Model : epistasis.models object
        Instance of an epistasis model.
    """
    def __init__(self, Classifier, Model):
        # Store model specs.
        self.model_specs = dict(Model=Model, Classifier=Classifier)
        self.Model = Model
        self.Classifier = Classifier
        self.Xbuilt = {}

    @property
    def parameters(self):
        """Nonlinear parameters"""
        return self.Model.parameters

    @property
    def epistasis(self):
        """High-order epistasis coefficients."""
        return self.Model.Linear.epistasis

    def add_gpm(self, gpm):
        """ Attach a GenotypePhenotypeMap object to the epistasis model.

        Also exposes APIs that are only accessible with a GenotypePhenotypeMap
        attached to the model.
        """
        self._gpm = gpm
        self.Model.add_gpm(gpm)
        self.Classifier.add_gpm(gpm)

    def fit(self, X='obs', y='obs',
            sample_weight=None,
            use_widgets=False,
            **kwargs):
        """Fit mixed model in two parts. 1. Use Classifier to predict the
        class of each phenotype (Dead/Alive). 2. Estimate the additive
        phenotypes on the full data set. 3. Fit scale and epistasis on the
        alive subset.

        Do to the nature of the mixed model, the fit method is less flexible
        than models in this package. This model requires that a
        GenotypePhenotypeMap object be attached.

        X must be:

        - 'obs' :
            Uses `gpm.binary` to construct X. If genotypes
            are missing they will not be included in fit. At the end of
            fitting, an epistasis map attribute is attached to the model
            class.
        - 'complete' :
            Uses `gpm.complete_binary` to construct X.
            All genotypes missing from the data are included. Warning, will
            break in most fitting methods. At the end of fitting, an
            epistasis map attribute is attached to the model class.
        - 'fit' :
            a previously defined array/dataframe matrix. Prevents
            copying for efficiency.

        y must be:

        - 'obs' :
            Uses `gpm.binary` to construct y. If
            phenotypes are missing they will not be included in fit.
        - 'complete' :
            Uses `gpm.complete_binary` to construct
            X. All genotypes missing from the data are included. Warning, will
            break in most fitting methods.
        - 'fit' :
            a previously defined array/dataframe matrix. Prevents
            copying for efficiency.

        Parameters
        ----------
        use_widgets : bool (default=False)
            If True, turns nonlinear parameters into ipywidgets.

        Keyword Arguments
        -----------------
        Keyword arguments are read as parameters to the nonlinear scale fit.
        """
        # Sanity checks on input.
        if hasattr(self, "gpm") is False:
            raise FittingError(
                "A GenotypePhenotypeMap must be attached to this model.")

        # Make sure X and y strings match
        if type(X) == str and type(y) == str and X != y:
            raise FittingError("Any string passed to X must be the same as "
                               "any string passed to y. "
                               "For example: X='obs', y='obs'.")

        # Else if both are arrays, check that X and y match dimensions.
        elif type(X) != str and type(y) != str and X.shape[0] != y.shape[0]:
            raise FittingError("X dimensions {} and y dimensions {} "
                               "don't match.".format(X.shape[0], y.shape[0]))

        # Handle y.

        # Get pobs for nonlinear fit.
        if type(y) is str and y in ["obs", "complete"]:
            y = self.gpm.phenotypes
        # Else, numpy array or dataframe
        elif type(y) == np.array or type(y) == pd.Series:
            pass
        else:
            raise FittingError("y is not valid. Must be one of the following:"
                               "'obs', 'complete', "
                               "numpy.array", "pandas.Series")

        # Handle X
        self.Classifier.fit(X=X, y=y)

        # Use model to infer dead phenotypes
        ypred = self.Classifier.predict(X="fit")
        yprob = self.Classifier.predict_proba(X="fit")

        # Build an X matrix for the Epistasis model.
        x = self.Model.add_X(X="obs")

        # Subset the data (and x matrix) to only include alive
        # genotypes/phenotypes
        y_subset = y[ypred == 1]
        # y_subset = y_subset.reset_index(drop=True)
        x_subset = x[ypred == 1, :]
        p_subset = yprob[ypred == 1, 1]

        # Fit model to the alive phenotype subset
        try:
            # For fitting nonlinear models
            self.Model._fit_additive(X=X, y=y, sample_weight=sample_weight)
            self.Model._fit_nonlinear(X=x_subset, y=y_subset)
            self.Model._fit_linear(X=x_subset, y=y_subset,
                                   sample_weight=sample_weight)

            # Otherwise fit linear model.
        except AttributeError:
            self.Model.fit(X=x_subset, y=y_subset,
                           sample_weight=sample_weight,
                           use_widgets=use_widgets, **kwargs)

        return self

    def plot_fit(self):
        """Plots the observed phenotypes against the additive model
        phenotypes"""
        padd = self.Additive.predict()
        pobs = self.gpm.phenotypes
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.plot(padd, pobs, '.b')
        plt.show()
        return fig, ax

    def predict(self, X='complete'):
        """Predict phenotypes given a model matrix. Constructs the predictions
        in two steps. 1. Use X to predict quantitative phenotypes. 2. Predict
        phenotypes classes using the Classifier. Xfit for the classifier is
        truncated to the order given by self.Classifier.order

        Return
        ------
        X : array
            Model matrix.
        ypred : array
            Predicted phenotypes.
        """
        # Predict the phenotypes from the epistasis model first.
        ypred = self.Model.predict(X=X)

        # Then predict the classes.
        yclasses = self.Classifier.predict(X=X)

        # Set any 0-class phenotypes to a value of 0
        ypred[yclasses == 0] = self.Classifier.threshold
        ypred[ypred < self.Classifier.threshold] = self.Classifier.threshold
        return ypred

    def score(self, X='obs', y='obs', sample_weight=None):
        """Calculates the squared-pearson coefficient for the nonlinear fit.

        Returns
        -------
        classifier_score : float
            score of classifier model.
        model_score : float
            squared pearson coefficient between phenotypes and nonlinear
            function.
        """
        # Sanity checks on input.
        if hasattr(self, "gpm") is False:
            raise FittingError(
                "A GenotypePhenotypeMap must be attached to this model.")

        # Make sure X and y strings match
        if type(X) == str and type(y) == str and X != y:
            raise FittingError("Any string passed to X must be the same as any"
                               "string passed to y. "
                               "For example: X='obs', y='obs'.")

        # Else if both are arrays, check that X and y match dimensions.
        elif type(X) != str and type(y) != str and X.shape[0] != y.shape[0]:
            raise FittingError("X dimensions {} and y dimensions"
                               "{} don't match.".format(X.shape[0],
                                                        y.shape[0]))

        # Handle y.
        # Get pobs for nonlinear fit.
        if type(y) is str and y in ["obs", "complete"]:
            pobs = self.gpm.phenotypes
        # Else, numpy array or dataframe
        elif type(y) == np.array or type(y) == pd.Series:
            pobs = y
        else:
            raise FittingError("y is not valid. Must be one of the "
                               "following: 'obs', 'complete', "
                               "numpy.array", "pandas.Series")

        # Use model to infer dead phenotypes
        classifier_score = self.Classifier.score(X=X, y=y)
        classes = self.Classifier.predict(X="fit")

        predictions = self.Model.predict(X=X)
        predictions = predictions[classes == 1]
        pobs = pobs[classes == 1]

        model_score = pearson(pobs, predictions)**2
        return classifier_score, model_score

    def contributions(self):
        """Calculate the contributions from each piece of the model.

        Returns a dictionary that includes Classifier and Model contribution.
        The Model contributions are ordered as additive, scale, and epistasis.
        """
        # Predict class
        pclass = self.Classifier.predict(X='fit')
        pclass2 = self.Classifier.predict()
        zero = pclass2[pclass2 == 0]
        # Zero-classes contribution
        class_contrib = len(zero) / len(self.gpm.complete_genotypes)

        # Calculate predicted phenotypes for each piece of model.
        model_contrib = self.Model.contributions()
        # # Quantitative phenotypes
        # x0 = self.gpm.phenotypes[pclass == 1]
        #
        # # Additive contribution.
        # x1 = self.Model.Additive.predict(X='fit')
        # x1 = x1[pclass == 1]
        #
        # # Scale contribution
        # x2 = self.Model.function(x1, **self.parameters, data=x1)
        #
        # # Epistasis contribution
        # x3 = self.Model.predict(X='fit')
        #
        # # Calculate contributions
        # additive = pearson(x0, x1)**2
        # scale = pearson(x0, x2)**2
        # epistasis = pearson(x0, x3)**2

        contributions = {'Classifier': class_contrib,
                         'Model': model_contrib}

        return contributions

    @property
    def thetas(self):
        """1d array of all coefs in model. The classifier coefficients are
        first in the array, then the model coefficients. See the thetas
        attributes of the input classifier/epistasis models to determine
        what is included in this combined array.
        """
        return np.concatenate((self.Classifier.thetas, self.Model.thetas))

    def hypothesis(self, X='obs', thetas=None):
        """Return a model's output with the given model matrix X and coefs."""
        # Use thetas to predict the probability of 1-class for each phenotype.
        if thetas is None:
            thetas = self.thetas

        # Sort thetas for classifier and model.
        thetas1 = thetas[0:len(self.Classifier.coef_[0])]
        thetas2 = thetas[len(self.Classifier.coef_[0]):]

        # 1. Class probability given the coefs
        proba = self.Classifier.hypothesis(X=X, thetas=thetas1)
        classes = np.ones(len(proba))
        classes[proba > 0.5] = 0

        # 2. Determine ymodel given the coefs.
        y = self.Model.hypothesis(X=X, thetas=thetas2)
        y[classes == 0] = self.Classifier.threshold
        y[y < self.Classifier.threshold] = self.Classifier.threshold
        return y

    def lnlike_of_data(self, X='obs', y='obs', yerr='obs',
                       sample_weight=None, thetas=None):
        """Calculate the log likelihood of data, given a set of model coefficients.

        Parameters
        ----------
        X : 2d array
            model matrix
        yerr: array
            uncertainty in data
        thetas : array
            array of model coefficients

        Returns
        -------
        lnlike : float
            log-likelihood of the data given the model.
        """
        if thetas is None:
            thetas = self.thetas

        thetas1 = thetas[0:len(self.Classifier.coef_[0])]
        thetas2 = thetas[len(self.Classifier.coef_[0]):]

        # Calculate log-likelihood of classifier
        class_lnlike = self.Classifier.lnlike_of_data(
            X=X, y=y, thetas=thetas1)
        classes = self.Classifier.predict(X=X)

        # Calculate log-likelihood of the model
        model_lnlike = self.Model.lnlike_of_data(X=X, y=y, thetas=thetas2)

        # Set the likelihoods of points below threshold to threshold
        model_lnlike[classes == 0] = 0

        # Sum the log-likelihoods
        return class_lnlike + model_lnlike
