# 
# import unittest
# import pytest
# from gpmap import GenotypePhenotypeMap
# from ..mixed import EpistasisMixedRegression
# 
# @pytest.fixture
# def gpm():
#     """Create a genotype-phenotype map"""
#     wildtype = "000"
#     genotypes =  ["000", "001", "010", "100", "011", "101", "110", "111"]
#     phenotypes = [  0.0,   0.2,   0.2,   0.4,   0.4,   0.8,   1.5,   2.0]
#     return GenotypePhenotypeMap(wildtype, genotypes, phenotypes)
#     
# class TestEpistasisMixedRegression(object):
# 
#     # Set some initial parameters for this model
#     order = 3
#     threshold = 0.1
#         
#     def test_init(self, gpm):
#         model = EpistasisMixedRegression(self.order, self.threshold)
#         assert hasattr(model, "Model")
#         assert hasattr(model, "Classifier")
#         assert hasattr(model, "order")
#         assert hasattr(model, "threshold")
#         assert hasattr(model, "model_type")
#     
#     def test_add_gpm(self, gpm):
#         model = EpistasisMixedRegression(self.order, self.threshold)
#         model.add_gpm(gpm)
#         assert hasattr(model, "gpm")
#         assert hasattr(model.Model, "gpm")
#         assert hasattr(model.Classifier, "gpm")
#     
#     def test_fit(self, gpm):
#         model = EpistasisMixedRegression(self.order, self.threshold)
#         model.add_gpm(gpm)
#         model.fit(lmbda=0, A=3,B=0)
#         assert hasattr(model, "Model")
#         
#     def test_predict(self, gpm):
#         model = EpistasisMixedRegression(self.order, self.threshold)
#         model.add_gpm(gpm)   
#         model.fit(lmbda=0, A=3,B=0)
#         predicted = model.predict(X="complete")
#         assert "predict" in model.Model.Linear.Xbuilt
#         assert "predict" in model.Model.Additive.Xbuilt
#         assert len(predicted) == gpm.n 
# 
#         
#         
#         
#         
