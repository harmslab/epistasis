import numpy as np
import pandas as pd
from sklearn.preprocessing import binarize
from sklearn.base import BaseEstimator, RegressorMixin

import epistasis.mapping
from epistasis.model_matrix_ext import get_model_matrix

from .base import BaseModel
from .power import EpistasisPowerTransform
from .classifiers import EpistasisLogisticRegression
from .utils import FittingError, XMatrixException


class EpistasisMixedRegression(BaseModel, BaseEstimator):
    """A high-order epistasis regression that first classifies genotypes as
    viable/nonviable (given some threshold) and then estimates epistatic
    coefficients in viable phenotypes.

    Parameters
    ----------
    order : int
        Order of epistasis in model
    threshold : float
        value below which phenotypes are considered dead
    model_type : str
        type of epistasis model to use.
    epistasis_model : epistasis.models object
        Epistasis model to use.
    epistasis_classifier : epistasis.models.classifier
        Epistasis classifier to use.

    Keyword Arguments
    -----------------
    Keyword arguments are interpreted as intial guesses for the nonlinear
    function parameters. Must have the same name as parameters in the
    nonlinear function.
    """

    def __init__(self, order, threshold, model_type="global",
                 epistasis_model=EpistasisPowerTransform,
                 epistasis_classifier=EpistasisLogisticRegression,
                 **p0):

        # Set model specs.
        self.order = order
        self.threshold = threshold
        self.model_type = model_type

        # Store model specs.
        self.model_specs = dict(
            order=self.order,
            threshold=self.threshold,
            model_type=self.model_type,
            epistasis_model=EpistasisPowerTransform,
            epistasis_classifier=EpistasisLogisticRegression,
            **p0)

        # Initialize the epistasis model
        self.Model = epistasis_model(order=self.order,
                                     model_type=self.model_type, **p0)

        # Initialize the epistasis classifier
        # Hardcode the classifier model as a first order model.
        classifier_order = 1

        self.Classifier = epistasis_classifier(
            threshold=self.threshold,
            order=classifier_order,
            model_type=self.model_type)

    def add_gpm(self, gpm):
        """ Attach a GenotypePhenotypeMap object to the epistasis model.

        Also exposes APIs that are only accessible with a GenotypePhenotypeMap
        attached to the model.
        """
        super(EpistasisMixedRegression, self).add_gpm(gpm)
        self.Model.add_gpm(gpm)
        self.Classifier.add_gpm(gpm)

    def fit(self, X='obs', y='obs', sample_weight=None, use_widgets=False, **kwargs):
        """Fit mixed model in two parts. 1. Use Classifier to predict the
        class of each phenotype (Dead/Alive). 2. Fit epistasis Model.

        Do to the nature of the mixed model, the fit method is less flexible than 
        models in this package. This model requires that a GenotypePhenotypeMap
        object be attached.

        X must be:

            - 'obs' : Uses `gpm.binary.genotypes` to construct X. If genotypes are missing
                they will not be included in fit. At the end of fitting, an epistasis map attribute
                is attached to the model class.
            - 'complete' : Uses `gpm.binary.complete_genotypes` to construct X. All genotypes
                missing from the data are included. Warning, will break in most fitting methods.
                At the end of fitting, an epistasis map attribute is attached to the model class.
            - 'fit' : a previously defined array/dataframe matrix. Prevents copying for efficiency.

        y must be:
            - 'obs' : Uses `gpm.binary.phenotypes` to construct y. If phenotypes are missing
                they will not be included in fit. 
            - 'complete' : Uses `gpm.binary.complete_genotypes` to construct X. All genotypes
                missing from the data are included. Warning, will break in most fitting methods.
            - 'fit' : a previously defined array/dataframe matrix. Prevents copying for efficiency.


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
            raise FittingError("Any string passed to X must be the same as any string passed to y. "
                               "For example: X='obs', y='obs'.")

        # Else if both are arrays, check that X and y match dimensions.
        elif type(X) != str and type(y) != str and X.shape[0] != y.shape[0]:
            raise FittingError("X dimensions {} and y dimensions {} don't match.".format(
                X.shape[0], y.shape[0]))

        # Handle y.

        # Get pobs for nonlinear fit.
        if type(y) is str and y in ["obs", "complete"]:
            pobs = self.gpm.binary.phenotypes
        # Else, numpy array or dataframe
        elif type(y) == np.array or type(y) == pd.Series:
            pobs = y
        else:
            raise FittingError("y is not valid. Must be one of the following: 'obs', 'complete', "
                               "numpy.array", "pandas.Series")

        # Handle X
        self.Classifier.fit(X=X, y=y)

        # Use model to infer dead phenotypes
        ypred = self.Classifier.predict(X="fit")

        # Build an X matrix for the Epistasis model.
        x = self.Model.add_X(X="obs")

        # Subset the data (and x matrix) to only include alive genotypes/phenotypes
        y_subset = pobs[ypred == 1]
        y_subset = y_subset.reset_index(drop=True)
        x_subset = x[ypred == 1, :]

        # Fit model to the alive phenotype supset
        out = self.Model.fit(
            X=x_subset, y=y_subset, sample_weight=sample_weight, use_widgets=use_widgets, **kwargs)

        return out

    def plot_fit(self):
        """Plots the observed phenotypes against the additive model phenotypes"""
        padd = self.Additive.predict()
        pobs = self.gpm.phenotypes
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.plot(padd, pobs, '.b')
        plt.show()
        return fig, ax

    def predict(self, X='complete'):
        """Predict phenotypes given a model matrix. Constructs the predictions in
        two steps. 1. Use X to predict quantitative phenotypes. 2. Predict phenotypes
        classes using the Classifier. Xfit for the classifier is truncated to the
        order given by self.Classifier.order

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
        ypred[yclasses == 0] = self.threshold
        ypred[ypred < self.threshold] = self.threshold
        return ypred

    def score(self, X='obs', y='obs', sample_weight=None):
        """Calculates the squared-pearson coefficient for the nonlinear fit.

        Returns
        -------
        r_nonlinear : float
            squared pearson coefficient between phenotypes and nonlinear function.
        r_linear : float
            squared pearson coefficient between linearized phenotypes and linear epistasis model
            described by epistasis.values.
        """
        # Sanity checks on input.
        if hasattr(self, "gpm") is False:
            raise FittingError(
                "A GenotypePhenotypeMap must be attached to this model.")

        # Make sure X and y strings match
        if type(X) == str and type(y) == str and X != y:
            raise FittingError("Any string passed to X must be the same as any string passed to y. "
                               "For example: X='obs', y='obs'.")

        # Else if both are arrays, check that X and y match dimensions.
        elif type(X) != str and type(y) != str and X.shape[0] != y.shape[0]:
            raise FittingError("X dimensions {} and y dimensions {} don't match.".format(
                X.shape[0], y.shape[0]))

        # Handle y.

        # Get pobs for nonlinear fit.
        if type(y) is str and y in ["obs", "complete"]:
            pobs = self.gpm.binary.phenotypes
        # Else, numpy array or dataframe
        elif type(y) == np.array or type(y) == pd.Series:
            pobs = y
        else:
            raise FittingError("y is not valid. Must be one of the following: 'obs', 'complete', "
                               "numpy.array", "pandas.Series")

        # Use model to infer dead phenotypes
        ypred = self.Classifier.predict(X="fit")

        # Subset the data (and x matrix) to only include alive genotypes/phenotypes
        y_subset = pobs[ypred == 1]
        y_subset = y_subset.reset_index(drop=True)

        scores = self.Model.score(
            X='fit', y=y_subset, sample_weight=sample_weight)
        return (self.Classifier.score(X=X, y=y),) + scores

    def contributions(self, X='obs', y='obs'):
        """Calculate the contributions from nonlinearity and epistasis to the variation in phenotype. 

        Returns
        -------
        contribs 
        """
        # Sanity checks on input.
        if hasattr(self, "gpm") is False:
            raise FittingError(
                "A GenotypePhenotypeMap must be attached to this model.")

        # Make sure X and y strings match
        if type(X) == str and type(y) == str and X != y:
            raise FittingError("Any string passed to X must be the same as any string passed to y. "
                               "For example: X='obs', y='obs'.")

        # Else if both are arrays, check that X and y match dimensions.
        elif type(X) != str and type(y) != str and X.shape[0] != y.shape[0]:
            raise FittingError("X dimensions {} and y dimensions {} don't match.".format(
                X.shape[0], y.shape[0]))

        # Handle y.

        # Get pobs for nonlinear fit.
        if type(y) is str and y in ["obs", "complete"]:
            pobs = self.gpm.binary.phenotypes
        # Else, numpy array or dataframe
        elif type(y) == np.array or type(y) == pd.Series:
            pobs = y
        else:
            raise FittingError("y is not valid. Must be one of the following: 'obs', 'complete', "
                               "numpy.array", "pandas.Series")

        # Use model to infer dead phenotypes
        ypred = self.Classifier.predict(X="fit")

        # Subset the data (and x matrix) to only include alive genotypes/phenotypes
        y_subset = pobs[ypred == 1]
        y_subset = y_subset.reset_index(drop=True)
        return self.Model.contributions(X='fit', y=y_subset)

        # scores = self.Model.score(X='fit', y=y_subset)
        #
        # # Calculate various pearson coeffs.
        # add_score = self.Additive.score()
        # scores = self.score(X=X, y=y)
        #
        # # Calculate the nonlinear contribution
        # nonlinear_contrib = scores[0] - add_score
        #
        # # Calculate the contribution from epistasis
        # epistasis_contrib = 1 - scores[0]
        #
        # # Build output dict.
        # contrib = {'nonlinear': nonlinear_contrib, 'epistasis': epistasis_contrib}
        # return contrib

    @property
    def thetas(self):
        """1d array of all coefs in model. The classifier coefficients are first
        in the array, then the model coefficients. See the thetas attributes of
        the input classifier/epistasis models to determine what is included in this
        combined array.
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
        y[classes == 0] = self.threshold
        y[y < self.threshold] = self.threshold
        return y

    def lnlike_of_data(self, X='obs', y='obs', yerr='obs', sample_weight=None, thetas=None):
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
