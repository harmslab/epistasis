
import unittest
import pytest
from gpmap import GenotypePhenotypeMap
from ..mixed import EpistasisMixedRegression

@pytest.fixture
def gpm():
    """Create a genotype-phenotype map"""
    wildtype = "000"
    genotypes =  ["000", "001", "010", "100", "011", "101", "110", "111"]
    phenotypes = [  0.0,   0.1,   0.5,   0.4,   0.2,   0.8,   0.5,   1.0]
    return GenotypePhenotypeMap(wildtype, genotypes, phenotypes)
    
class TestEpistasisMixedRegression(object):

    # Set some initial parameters for this model
    order = 3
    threshold = 0.2
        
    def test_add_gpm(self, gpm):
        model = EpistasisMixedRegression(self.order, self.threshold)
        model.add_gpm(gpm)
        assert hasattr(model, "gpm")
    
    def test_fit(self, gpm):
        model = EpistasisMixedRegression(self.order, self.threshold)
        model.add_gpm(gpm)
        model.fit()
        assert hasattr(model, "Xfit")
        
    def test_predict(self, gpm):
        model = EpistasisMixedRegression(self.order, self.threshold)
        model.add_gpm(gpm)   
        model.fit()     
        model.predict()
        assert hasattr(model, "Xpredict")
        
        
        
        
