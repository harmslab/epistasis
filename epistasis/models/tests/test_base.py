__doc__ = """Test module for gpm module. Tests construction, initialization,
and functionality of the GenotypePhenotypeMap object.
"""

import os, json

from gpmap.simulate import GenotypePhenotypeSimulation
from ..base import BaseModel

from nose import tools

def test_BaseModel_from_gpm():
    gpm = GenotypePhenotypeSimulation.from_length(3)
    model = BaseModel.from_gpm(gpm)
    tools.assert_true(hasattr(model, "gpm"))

def test_BaseModel_fit_raises_subclass_exception():
    model = BaseModel()
    tools.assert_raises(Exception, model.fit)

def test_BaseModel_predict_raises_subclass_exception():
    model = BaseModel()
    tools.assert_raises(Exception, model.predict)
