import pytest

# External imports
import numpy as np
from gpmap import GenotypePhenotypeMap

# Module to test
from ..classifiers import *

@pytest.fixture
def gpm():
    """Create a genotype-phenotype map"""
    wildtype = "000"
    genotypes =  ["000", "001", "010", "100", "011", "101", "110", "111"]
    phenotypes = [  0.0,   0.1,   0.5,   0.4,   0.2,   0.8,   0.5,   1.0]
    return GenotypePhenotypeMap(wildtype, genotypes, phenotypes)

class TestEpistasisLogisticRegression(object):
    
    # Set some initial parameters for this model
    order = 1
    threshold = 0.2

    def test_init(self, gpm):
        model = EpistasisLogisticRegression.read_gpm(gpm, threshold=self.threshold, order=self.order, model_type="local")
        # Checks
        check1 = model.order
        check2 = model.model_type
        # Tests
        assert check1 == self.order
        assert check2 == "local"

    # def test_compare_proba_to_hypothesis():
    #     gpm = GenotypePhenotypeSimulation.from_length(2)
    #     gpm.phenotypes = np.array([0, 0.1, 0.5, 1])
    #     model = EpistasisLogisticRegression.read_gpm(gpm, threshold=.2, order=1, model_type="global")
    #     model.fit()
    #     # Two arrays to test
    #     proba = model.predict_proba()[:,1]
    #     hypothesis = model.hypothesis(thetas=model.epistasis.values)
    #     # Test
    #     np.testing.assert_array_almost_equal(proba, hypothesis)
