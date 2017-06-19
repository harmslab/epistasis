import numpy as np
from sklearn.preprocessing import binarize

import epistasis.mapping
from epistasis.model_matrix_ext import get_model_matrix

from .base import BaseModel
from .linear import EpistasisLinearRegression
from .classifiers import EpistasisLogisticRegression

# Suppress an annoying error
import warnings
warnings.filterwarnings(action="ignore", category=RuntimeWarning)


class EpistasisMixedRegression(BaseModel):
    """A generalized, mixed epistasis model. This mixes an epistasis
    classification and regression model, allowing you to pre-categorize dead/alive
    phenotypes in your data. It then removes dead phenotypes from the epistasis
    calculation, but uses those dead phenotypes to predict other dead phenotypes.
    """
    def __init__(self, order, threshold, model_type="global",
        epistasis_model=EpistasisLinearRegression,
        epistasis_classifier=EpistasisLogisticRegression,
        **kwargs):

        # Set model specs.
        self.order = order
        self.threshold = threshold
        self.model_type = model_type

        # Initialize the epistasis model
        self.Model = epistasis_model(order=self.order,
            model_type=self.model_type)

        # Initialize the epistasis classifier
        self.Classifier = epistasis_classifier(
            threshold=self.threshold,
            order=1,
            model_type=self.model_type)

    def attach_gpm(self, gpm):
        """ Attach a GenotypePhenotypeMap object to the epistasis model.

        Also exposes APIs that are only accessible with a GenotypePhenotypeMap
        attached to the model.
        """
        super(EpistasisMixedRegression, self).attach_gpm(gpm)
        self.Model.attach_gpm(gpm)
        self.Classifier.attach_gpm(gpm)

    def fit(self, X=None, y=None, **kwargs):
        """Fit mixed model in two parts. 1. Use Classifier to predict the
        class of each phenotype (Dead/Alive). 2. Fit epistasis Model.

        If X and y are given, Epistasis maps are not attached to the models.

        Parameters
        ----------
        X : 2d array
            epistasis model matrix.
        y : 1d array
            Phenotype array.

        Returns
        -------
        self : instance of EpistasisMixedLinearRegression
        """
        if y is None:
            y = self.gpm.phenotypes

        if X is None:
            # --------------------------------------------------------
            # Part 1: classify
            # --------------------------------------------------------
            # Build X matrix for classifier
            order = 1
            sites = epistasis.mapping.mutations_to_sites(order, self.gpm.mutations)
            Xclass = get_model_matrix(self.gpm.binary.genotypes, sites, model_type=self.model_type)

            # Fit classifier
            self.Classifier.fit(X=Xclass, y=y)

            # Append epistasis map to coefs
            self.Classifier.epistasis = epistasis.mapping.EpistasisMap(sites,
                order=order, model_type=self.model_type)
            self.Classifier.epistasis.values = self.Classifier.coef_.reshape((-1,))
            ypred = self.Classifier.predict(X=Xclass)

            # --------------------------------------------------------
            # Part 2: fit epistasis
            # --------------------------------------------------------
            # Build X matrix for epistasis model
            order = self.order
            sites = epistasis.mapping.mutations_to_sites(order, self.gpm.mutations)
            X = get_model_matrix(self.gpm.binary.genotypes, sites, model_type=self.model_type)

            # Ignore phenotypes that are found "dead"
            y = y[ypred==1]
            X = X[ypred==1,:]

            # Fit model
            self.Model.fit(X=X, y=y, **kwargs)

            # Append epistasis map to coefs
            self.Model.epistasis = epistasis.mapping.EpistasisMap(sites,
                order=order, model_type=self.model_type)
            self.Model.epistasis.values = self.Model.coef_.reshape((-1,))

        else:
            # --------------------------------------------------------
            # Part 1: classify
            # --------------------------------------------------------
            self.Classifier.fit()
            ypred = self.Classifier.predict(X=self.Classifier.Xfit)

            # Ignore phenotypes that are found "dead"
            y = y[ypred==1]
            X = X[ypred==1,:]

            # --------------------------------------------------------
            # Part 2: fit epistasis
            # --------------------------------------------------------
            self.Model.fit(X=X, y=y, **kwargs)

        return self

    def predict(self, X=None):
        """Predict phenotypes given a model matrix. Constructs the predictions in
        two steps. 1. Use X to predict quantitative phenotypes. 2. Predict phenotypes
        classes using the Classifier. Xfit for the classifier is truncted to the
        order given by self.Classifier.order

        Return
        ------
        X : array
            Model matrix.
        ypred : array
            Predicted phenotypes.
        """
        if X is None:
            # 1. Predict quantitative phenotype
            ypred = self.Model.predict()
            # 2. Determine class (Dead/alive) for each phenotype
            yclasses = self.Classifier.predict()
            # Update ypred with dead phenotype information
            ypred[yclasses==0] = 0
        else:
            # 1. Predict quantitative phenotype
            ypred = self.Model.predict(X=X)
            # 2. Determine class (Dead/alive) for each phenotype
            nterms = self.Classifier.Xfit.shape[-1]
            Xclass = X[:,:nterms]
            yclasses = self.Classifier.predict(X=Xclass)
            # Update ypred with dead phenotype information
            ypred[yclasses==0] = 0

        return ypred

    def hypothesis(self, X=None, thetas=None):
        """Return a model's output with the given model matrix X and coefs."""
        # Use thetas to predict the probability of 1-class for each phenotype.
        if thetas is None:
            thetas = self.thetas

        thetas1 = thetas[0:len(self.Classifier.coef_[0])]
        thetas2 = thetas[len(self.Classifier.coef_[0]):]

        # 1. Class probability given the coefs
        proba = self.Classifier.hypothesis(thetas=thetas1)
        classes = np.ones(len(proba))
        classes[proba<0.5] = 0

        # 2. Determine ymodel given the coefs.
        y = self.Model.hypothesis(thetas=thetas2)
        y = np.multiply(y, classes)
        return y

    def lnlikelihood(self, ydata=None, yerr=None, thetas=None):
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
        ymodel : array
            predicted output from model.
        """
        ### Data
        if ydata is None:
            ydata = self.gpm.phenotypes
            yerr = self.gpm.std.upper
        # Binarize the data
        ybin = binarize(ydata, threshold=self.threshold)[0]#np.ones(len(y_class_prob))

        if thetas is None:
            thetas = self.thetas

        thetas1 = thetas[0:len(self.Classifier.coef_[0])]
        thetas2 = thetas[len(self.Classifier.coef_[0]):]

        # 1. Class probability given the coefs
        y_class_prob = self.Classifier.hypothesis(thetas=thetas1)
        classes = np.ones(len(y_class_prob))
        classes[y_class_prob<0.5] = 0

        # 2. Determine ymodel given the coefs.
        ymodel = self.Model.hypothesis(thetas=thetas2)
        ymodel = np.multiply(ymodel, classes)

        ### log-likelihood of logit model
        lnlikelihood = ybin * np.log(y_class_prob) + (1 - ybin) * np.log(1-y_class_prob)

        ### log-likelihood of the epistasis model
        inv_sigma2 = 1.0/(yerr**2)
        lngaussian = (ydata-ymodel)**2*inv_sigma2 - np.log(inv_sigma2)
        lnlikelihood[ybin==1] = np.add(lnlikelihood[ybin==1], lngaussian[ybin==1])
        return -0.5 * sum(lnlikelihood), ymodel

    @property
    def thetas(self):
        """1d array of all coefs in model."""
        return np.concatenate((self.Classifier.thetas, self.Model.thetas))
