__doc__ = """Test module for gpm module. Tests construction, initialization,
and functionality of the GenotypePhenotypeMap object.
"""

import os, json
import pytest
from gpmap import GenotypePhenotypeMap
from ..base import BaseModel

@pytest.fixture
def gpm():
    """Create a genotype-phenotype map"""
    wildtype = "000"
    genotypes =  ["000", "001", "010", "100", "011", "101", "110", "111"]
    phenotypes = [  0.0,   0.1,   0.5,   0.4,   0.2,   0.8,   0.5,   1.0]
    return GenotypePhenotypeMap(wildtype, genotypes, phenotypes)

class TestBaseModel():
    
    def test_read_gpm(self, gpm):
        model = BaseModel.read_gpm(gpm)
        assert hasattr(model, "gpm") == True

    def test_fit(self, gpm):
        model = BaseModel()
        with pytest.raises(Exception):
            model.fit()

    def test_predict(self, gpm):
        model = BaseModel()
        with pytest.raises(Exception):
            model.predict()
