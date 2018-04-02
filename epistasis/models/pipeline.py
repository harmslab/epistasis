"""Epistasis Pipeline module."""

import numpy as np
from ..stats import pearson
from .base import BaseModel

class EpistasisPipeline(list, BaseModel):
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

    def fit(self, X=None, y=None):
        # Fit the first model
        model = self[0]
        gpm = model.fit_transform(X=X, y=y)

        # Then fit every model afterwards.
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

    def predict(self, X=None):
        # Predict from last model in the list first.
        model = self[-1]
        ypred = model.predict_transform(X=X, y=y)

        # Then work backwards predicting/transforming until the first model.
        for model in self[-2::-1]:
            ypred = model.predict_transform(X=X, y=ypred)

        # Return predictions
        return ypred

    def hypothesis(self, X=None, y=None, thetas=None):
        pass

    def hypothesis_transform(self, X=None, y=None, thetas=None):
        pass

    def score(self, X=None, y=None):
        ypred = self.predict(X=X)
        return pearson(y, ypred)**2

    @property
    def thetas(self):
        # All parameters in order of models.
        thetas = [m.thetas for m in self]
        return np.concatenate(thetas)

    def lnlike_of_data(
            self,
            X=None,
            y=None,
            yerr=None,
            thetas=None):

        # Calculate likelihood of data.
        ymodel = self.hypothesis(X=X, y=y, thetas=thetas)
        L = - 0.5 * np.log(2 * np.pi * yerr**2) - (0.5 * ((y - ymodel)**2 / yerr**2))
        return L
