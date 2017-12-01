# #from gpmap.simulate import GenotypePhenotypeSimulation
# from ..utils import *
#
# from ..base import BaseModel
#
# class MockModel(BaseModel):
#
#     def __init__(self):
#         self.gpm = GenotypePhenotypeSimulation.from_length(2)
#         self.model_type = "local"
#         self.order = 2
#         self.Xbuilt = {}
#
#     @X_fitter
#     def fit(self, X='obs', y='obs'):
#         self.coef_ = [0,0,0,0]
#         return None
#
#     @X_predictor
#     def predict(self, X='complete', y='complete'):
#         return None
#
# def test_X_fitter():
#     model = MockModel()
#     model.fit()
#     # Test an Xfit matrix was made
#     assert "obs" in model.Xbuilt
#     assert "fit" in model.Xbuilt
#     assert  model.Xbuilt["fit"].shape == (4,4)
#
# def test_X_predictor():
#     model = MockModel()
#     model.fit()
#     model.predict()
#     # Test an Xfit matrix was made
#     assert "complete" in model.Xbuilt
#     assert "predict" in model.Xbuilt
#     assert model.Xbuilt["predict"].shape == (4,4)
