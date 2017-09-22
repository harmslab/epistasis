# Externel imports
import pytest

import numpy as np
from gpmap import GenotypePhenotypeMap

# Module to test
from ..nonlinear import *


@pytest.fixture
def gpm():
    """Create a genotype-phenotype map"""
    wildtype = "000"
    genotypes =  ["000", "001", "010", "100", "011", "101", "110", "111"]
    phenotypes = [  0.0,   0.1,   0.5,   0.4,   0.2,   0.8,   0.5,   1.0]
    return GenotypePhenotypeMap(wildtype, genotypes, phenotypes)


def function(x, A, B):
    return A*x + B 

def reverse(y, A, B):
    return (y - B) / A


class TestEpistasisNonlinearRegression(object):

    order = 3
    model_type = "local"
    
    
    def test_init(self, gpm):
        model = EpistasisNonlinearRegression.read_gpm(gpm,
            function=function,
            reverse=reverse,
            order=self.order,
            model_type=self.model_type)

        # Checks
        assert hasattr(model, 'parameters') == True
        assert hasattr(model, 'function') == True
        assert hasattr(model, 'reverse') == True
                
        parameters = model.parameters
            
        # Checks
        assert hasattr(parameters, 'A') == True
        assert hasattr(parameters, 'B') == True

    def test_fit(self, gpm):

        model = EpistasisNonlinearRegression.read_gpm(gpm,
            function=function,
            reverse=reverse,
            order=self.order,
            model_type=self.model_type)
        
        model.fit(A=1, B=0)
        
        assert hasattr(model.Linear, 'Xbuilt') == True
        assert "fit" in model.Linear.Xbuilt

    def test_predict(self, gpm):
        model = EpistasisNonlinearRegression.read_gpm(gpm,
            function=function,
            reverse=reverse,
            order=self.order,
            model_type=self.model_type)
        
        model.fit(A=1, B=0)
        y = model.predict()
        
        # Tests
        np.testing.assert_almost_equal(sorted(y), sorted(model.gpm.phenotypes))
    # 
    # def test_thetas(self, gpm):
    #     model = EpistasisNonlinearRegression.read_gpm(gpm,
    #         function=function,
    #         reverse=reverse,
    #         order=self.order,
    #         model_type=self.model_type)
    #     
    #     model.fit(A=1,B=0)
    #     coefs = model.thetas
    #     # Tests
    #     assert len(coefs) == 6
    # 
    # def test_hypothesis(self, gpm):
    # 
    #     model = EpistasisNonlinearRegression.read_gpm(gpm,
    #         function=function,
    #         reverse=reverse,
    #         order=self.order,
    #         model_type=self.model_type)
    #     
    #     model.fit(A=1,B=0)
    #     predictions = model.hypothesis()
    #     # Tests
    #     np.testing.assert_almost_equal(sorted(predictions), sorted(model.gpm.phenotypes))
