from .classifiers import EpistasisLogisticRegression
from sklearn.preprocessing import binarize

class ModelPreprocessor(object):
    """Adds a preprocessing classifier to other epistasis models
    """
    def classify_gpm(self, threshold):
        """Classify genotype-phenotypes as dead or alive given some minimum threshold.
        Alive phenotypes are used to fit the model coefficients. Dead phenotypes
        are
        """
        # Classify phenotypes into dead or alive.
        setattr(self, "get_mutation_lethality", self._get_mutation_lethality)
        self.Classifier = EpistasisLogisticRegression.from_gpm(self.gpm,
                threshold=threshold,
                order=1,
                model_type=self.model_type).fit()
        return self

    def _get_mutation_lethality(self):
        return EpistasisLogisticRegression.coefs_
