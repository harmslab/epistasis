"""Epistasis Pipeline module."""

from .stats import pearson

class EpistasisPipeline(object):
    """Construct a pipeline of epistasis models to run in series.

    The models fit in order. A `fit_transform` method is called on each model
    one at a time. This method returns as transformed GenotypePhenotypeMap
    object and adds it to the next Epistasis model in the list.
    """
    def __init__(self, *models):
        self.models = models

    def __str__(self):
        s = "EpistasisPipeline(\n"
        for model in self.models:
            s += "    {},\n".format(model.__repr__())
        s += ")"
        return s

    def __repr__(self):
        return self.__str__()

    def add_gpm(self, gpm):
        self._gpm = gpm
        self.models[0].add_gpm(gpm)
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
        model = self.models[0]
        gpm = model.fit_transform(X=X, y=y)

        for model in self.models[1:]:
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

        model = self.models[-1]
        ypred = model.predict_transform(X=X)

        for model in self.models[-2::-1]:
            ypred = model.predict_transform(X=X, y=ypred)

        return ypred

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
