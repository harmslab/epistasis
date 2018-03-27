"""Epistasis Pipeline module."""

import numpy as np
from .stats import pearson

class EpistasisPipeline(list):
    """Construct a pipeline of epistasis models to run in series.

    This object is a subclass of a Python list. This means, all objects are
    changed in-place. This also means, you can append, prepend, remove,
    and rearrage Pipelines as you would a list.

    The models fit in order. A `fit_transform` method is called on each model
    one at a time. This method returns as transformed GenotypePhenotypeMap
    object and adds it to the next Epistasis model in the list.
    """
    @property
    def num_of_params(self):
        """"""
        return sum([self.num_of_params for m in self])

    def add_gpm(self, gpm):
        self._gpm = gpm
        self[0].add_gpm(gpm)
        return self

    @property
    def gpm(self):
        return self._gpm

    @gpm.setter
    def gpm(self, gpm):
        raise Exception("Can not add gpm directly. Use `add_gpm` method.")

    def fit(self, X='obs', y='obs'):
        """Fit pipeline."""
        # Fit the first model
        model = self[0]
        gpm = model.fit_transform(X=X, y=y)

        for model in self[1:]:
            # Get transformed genotype from modeli.
            model.add_gpm(gpm)

            # Fit model.
            try:
                gpm = model.fit_transform(X=X, y=y)
            except Exception as e:
                print("Failed with {}".format(model))
                print("Input was :")
                print("X : {}".format(X),
                print("y : {}".format(y)))
                raise e

        return self

    def predict(self, X='obs'):
        """Predict using pipeline."""
        if isinstance(X, str):
            X = self.gpm.genotypes
            y = self.gpm.phenotypes

        model = self[-1]
        ypred = model.predict_transform(X=X, y=y)

        for model in self[-2::-1]:
            ypred = model.predict_transform(X=X, y=ypred)

        return ypred

    def hypothesis(self, X='obs', y='obs', thetas=None):
        """"""
        if thetas is None:
            thetas = self.thetas





    def lnlike_of_data(self, ):
        """"""

    def lnlikelihood(self, ):
        """"""

    def score(self, X='obs', y='obs'):
        """Calculate pearson coefficient for between model and data.

        Note: if one of the models is a classifier, this number is likely to be
        a poor estimate. Genotypes classified as 0 will be included in the score
        which isn't correct. They should be ignored. Try dropping these Genotypes
        from your input and rerun this method.
        """
        if isinstance(y, str) and y == 'obs':
            y = self.gpm.phenotypes
        ypred = self.predict(X=X)
        return pearson(y, ypred)**2

    @property
    def thetas(self):
        """All parameters across all models in order of the models."""
        thetas = [m.thetas for m in self]
        return np.concatenate(thetas)

    def lnlike_of_data(
            self,
            X="obs",
            y="obs",
            yerr="obs",
            thetas=None
        ):
        """"""
        if thetas is None:
            ymodel = self.predict(X=X)

        if isinstance(y, str) and y == 'obs':
            y = self.gpm.phenotypes

            # Have to assume identical error, so we average errors.
            # May need to return to this,...
            yerr = np.mean(self.gpm.stdeviations)

        L = - 0.5 * np.log(2 * np.pi * yerr**2) - (0.5 * ((y - ymodel)**2 / yerr**2))
        return L

    def lnlikelihood(
            self,
            X="obs",
            y="obs",
            yerr="obs",
            thetas=None
        ):
        """Calculate the log likelihood of y, given a set of model coefficients.

        Parameters
        ----------
        X : 2d array
            model matrix
        y : array
            data to calculate the likelihood
        yerr: array
            uncertainty in data
        thetas : array
            array of model coefficients

        Returns
        -------
        lnlike : float
            log-likelihood of data given a model.
        """

        lnlike = np.sum(self.lnlike_of_data(X=X, y=y, yerr=yerr,
                                            thetas=thetas))

        # If log-likelihood is infinite, set to negative infinity.
        if np.isinf(lnlike) or np.isnan(lnlike):
            return -np.inf
        return lnlike
