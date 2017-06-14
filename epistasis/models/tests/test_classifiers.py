# External imports
import numpy as np
from nose import tools
from gpmap.simulate import GenotypePhenotypeSimulation

# Module to test
from ..classifiers import *

def test_EpistasisLogisticRegression_initialization():
    gpm = GenotypePhenotypeSimulation.from_length(2)
    model = EpistasisLogisticRegression.from_gpm(gpm, threshold=.1, order=1, model_type="local")
    # Checks
    check1 = model.order
    check2 = model.model_type
    # Tests
    tools.assert_equals(check1, 1)
    tools.assert_equals(check2, "local")

def test_EpistasisLogisticRegression_model_matrices_match():
    gpm = GenotypePhenotypeSimulation.from_length(2)
    gpm.phenotypes = np.array([0, 0.1, 0.5, 1])
    model = EpistasisLogisticRegression.from_gpm(gpm, threshold=.2, order=1, model_type="global")
    model.fit()
    model.predict()
    # Test
    np.testing.assert_array_equal(model.Xfit, model.Xpredict)

def test_EpistasisLogisticRegression_compare_proba_to_hypothesis():
    gpm = GenotypePhenotypeSimulation.from_length(2)
    gpm.phenotypes = np.array([0, 0.1, 0.5, 1])
    model = EpistasisLogisticRegression.from_gpm(gpm, threshold=.2, order=1, model_type="global")
    model.fit()
    # Two arrays to test
    proba = model.predict_proba()[:,0]
    hypothesis =model.hypothesis(thetas=model.epistasis.values)
    # Test
    np.testing.assert_array_almost_equal(proba, hypothesis)
