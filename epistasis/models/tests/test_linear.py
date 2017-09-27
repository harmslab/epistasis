# External imports
import unittest
import pytest

import numpy as np
from gpmap.simulate import GenotypePhenotypeSimulation

# Module to test
from ..linear import *

@pytest.fixture
def gpm():
    """Create a genotype-phenotype map"""
    wildtype = "000"
    genotypes =  ["000", "001", "010", "100", "011", "101", "110", "111"]
    phenotypes = [  0.1,   0.1,   0.5,   0.4,   0.2,   0.8,   0.5,   1.0]
    stdeviations = 0.1
    return GenotypePhenotypeMap(wildtype, genotypes, phenotypes, stdeviations=stdeviations)

class testEpistasisLinearRegression(object):
    
    order = 2

    def test_init(self, gpm):
        model = EpistasisLinearRegression.read_gpm(gpm, order=self.order, model_type="local")
        # Checks
        check1 = model.order
        check2 = model.model_type
        assert check1 == self.orders
        assert check2 == "local"

    def test_fit(self, gpm):
        model = EpistasisLinearRegression.read_gpm(gpm, order=self.order, model_type="local")
        model.fit()
        # Checks
        check1 = hasattr(model, "Xbuilt")
        check2 = hasattr(model, "coef_")
        check3 = hasattr(model, "epistasis")
        
        # Tests
        assert check1 == True
        assert check2 == True
        assert check3 == True
        assert "obs" in model.Xbuilt
        assert "fit" in model.Xbuilt

    def test_predict(self, gpm):
        model = EpistasisLinearRegression.read_gpm(gpm, order=self.order, model_type="local")
        model.fit()
        check1 = model.predict()

        # Tests
        np.testing.assert_almost_equal(sorted(check1), sorted(model.gpm.phenotypes))
        assert "predict" in model.Xbuilt
        assert "complete" in model.Xbuilt

    def test_score(self, gpm):
        model = EpistasisLinearRegression.read_gpm(gpm, order=self.order, model_type="local")
        model.fit()
        score = model.score()
        # Tests
        assert score >= 0 
        assert score <= 1

    def test_hypothesis(self, gpm):
        model = EpistasisLinearRegression.read_gpm(gpm, order=self.order, model_type="local")
        model.fit()
        # Checks
        check1 = model.hypothesis(thetas=model.coef_)
        # Tests
        np.testing.assert_almost_equal(sorted(check1), sorted(model.gpm.phenotypes))        
        
    def test_lnlikelihood(self, gpm):
        model = EpistasisLinearRegression.read_gpm(gpm, order=self.order, model_type="local")
        model.fit()
        
        # Calculate lnlikelihood
        lnlike = model.lnlikelihood()
        assert lnlike.dtype == float
