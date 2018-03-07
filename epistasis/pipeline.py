"""Epistasis Pipeline module."""

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

        model = self[-1]
        ypred = model.predict_transform(X=X)

        for model in self[-2::-1]:
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
