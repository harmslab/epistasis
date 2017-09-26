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
        # Tests
        assert model.order == self.order
        assert model.model_type == "local"

    def test_fit(self, gpm):
        model = EpistasisLogisticRegression.read_gpm(gpm, threshold=self.threshold, order=self.order, model_type="local")
        model.fit()
        
        assert "fit" in model.Xbuilt
        assert "obs" in model.Xbuilt
        assert hasattr(model, "classes")
        assert hasattr(model, "epistasis")
        assert hasattr(model, "coef_")
        
    def test_predict(self, gpm):
        model = EpistasisLogisticRegression.read_gpm(gpm, threshold=self.threshold, order=self.order, model_type="local")
        model.fit()
        ypred = model.predict()
        
        assert "predict" in model.Xbuilt
        assert "complete" in model.Xbuilt
        assert len(ypred) == model.gpm.n

    def test_predict_proba(self, gpm):
        model = EpistasisLogisticRegression.read_gpm(gpm, threshold=self.threshold, order=self.order, model_type="local")
        model.fit()
        probs = model.predict_proba()
        
        # check probs is the right length
        assert len(probs) == model.gpm.n
        
        # Check probs are between 0 and 1
        assert np.all(probs<=1)
        assert np.all(probs>=0)

    def test_predict_log_proba(self, gpm):
        model = EpistasisLogisticRegression.read_gpm(gpm, threshold=self.threshold, order=self.order, model_type="local")
        model.fit()
        probs = model.predict_log_proba()
        
        # check probs is the right length
        assert len(probs) == model.gpm.n
        
        # Check log probs are less than or equal to 0
        assert np.all(probs<=0)
        
    def test_score(self, gpm):
        model = EpistasisLogisticRegression.read_gpm(gpm, threshold=self.threshold, order=self.order, model_type="local")
        model.fit()
        score = model.score()    
        
        # Test score is between 0 and 1
        assert 0 <= score <= 1

    def test_thetas(self, gpm):
        model = EpistasisLogisticRegression.read_gpm(gpm, threshold=self.threshold, order=self.order, model_type="local")
        model.fit()
        
        # Check thetas is the correct length
        assert len(model.thetas) == len(model.coef_[0])

    def test_hypothesis(self, gpm):
        model = EpistasisLogisticRegression.read_gpm(gpm, threshold=self.threshold, order=self.order, model_type="local")
        model.fit()
        
        # these should be equal if working properly
        pred = model.predict_proba()[:,0]
        hypo = model.hypothesis()
        np.testing.assert_almost_equal(pred, hypo)
                
    def test_lnlikelihood(self, gpm):
        model = EpistasisLogisticRegression.read_gpm(gpm, threshold=self.threshold, order=self.order, model_type="local")
        model.fit()
        lnlike = model.lnlikelihood()

        # Check we get a float
        assert lnlike.dtype == float
