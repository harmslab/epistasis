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
        return EpistasisLogisticRegression.from_gpm(self.gpm,
                threshold=threshold,
                order=1,
                model_type=self.model_type).fit()
