from gpmap.simulate import GenotypePhenotypeSimulation
from ..utils import *

from nose import tools

class MockModel(object):

    def __init__(self):
        self.gpm = GenotypePhenotypeSimulation.from_length(2)
        self.model_type = "local"
        self.order = 2

    @X_fitter
    def fit(self, X=None, y=None):
        self.coef_ = [0,0,0,0]
        return None

    @X_predictor
    def predict(self, X=None, y=None):
        return None

def test_X_fitter_decorator_sets_Xfit_attribute():
    model = MockModel()
    model.fit()
    # Test an Xfit matrix was made
    check = hasattr(model, "Xfit")
    tools.assert_true(check)

def test_X_fitter_decorator_sets_Xfit_to_proper_shape():
    model = MockModel()
    model.fit()
    # Test an Xfit matrix was made
    check = model.Xfit.shape
    tools.assert_equals((4,4), check)

def test_X_predictor_decorator_sets_Xpredict_attribute():
    model = MockModel()
    model.fit()
    model.predict()
    # Test an Xfit matrix was made
    check = hasattr(model, "Xpredict")
    tools.assert_true(check)

def test_X_predictor_decorator_sets_Xpredict_to_proper_shape():
    model = MockModel()
    model.fit()
    model.predict()
    # Test an Xfit matrix was made
    check = model.Xpredict.shape
    tools.assert_equals((4,4), check)
