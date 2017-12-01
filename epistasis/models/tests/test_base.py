__doc__ = """Test module for gpm module. Tests construction, initialization,
and functionality of the GenotypePhenotypeMap object.
"""

import os
import json
import pytest
from gpmap import GenotypePhenotypeMap
from ..base import BaseModel, XMatrixException
from ...mapping import EpistasisMap


@pytest.fixture
def gpm():
    """Create a genotype-phenotype map"""
    wildtype = "000"
    genotypes = ["000", "001", "010", "100", "011", "101", "110", "111"]
    phenotypes = [0.0,   0.1,   0.5,   0.4,   0.2,   0.8,   0.5,   1.0]
    return GenotypePhenotypeMap(wildtype, genotypes, phenotypes)


class TestBaseModel():

    def test_add_gpm(self, gpm):
        model = BaseModel()
        model.add_gpm(gpm)

        assert hasattr(model, "gpm")

    def test_add_epistasis(self, gpm):
        model = BaseModel()
        model.order = 2
        model.model_type = "local"
        model.add_gpm(gpm)
        model.add_epistasis()

        assert hasattr(model, "epistasis")
        assert type(model.epistasis) == EpistasisMap

    def test_add_X(self, gpm):
        model = BaseModel()

        # Check that calling add_X before a gpm is attaches raises an error
        with pytest.raises(XMatrixException):
            model.add_X(X="obs")

        # Check that add_X works with each type of X
        model.add_gpm(gpm)
        model.order = 3
        model.model_type = "local"
        model.add_X(X="obs")

        assert "obs" in model.Xbuilt
        assert model.Xbuilt["obs"].shape == (8, 8)

    def test_fit(self, gpm):
        model = BaseModel()
        with pytest.raises(Exception):
            model.fit()

    def test_predict(self, gpm):
        model = BaseModel()
        with pytest.raises(Exception):
            model.predict()

    def test_hypothesis(self, gpm):
        model = BaseModel()
        with pytest.raises(Exception):
            model.hypothesis()

    def test_lnlikelihood(self, gpm):
        model = BaseModel()
        with pytest.raises(Exception):
            model.lnlikelihood()

    def test_read_gpm(self, gpm):
        model = BaseModel.read_gpm(gpm)
        assert hasattr(model, "gpm") is True
