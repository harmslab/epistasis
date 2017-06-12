# External imports
import numpy as np
from nose import tools
from gpmap.simulate import GenotypePhenotypeSimulation

# Module to test
from ..linear import *

def test_EpistasisLinearRegression_initialization():
    gpm = GenotypePhenotypeSimulation.from_length(2)
    model = EpistasisLinearRegression.from_gpm(gpm, order=2, model_type="local")
    # Checks
    check1 = model.order
    check2 = model.model_type
    tools.assert_equals(check1, 2)
    tools.assert_equals(check2, "local")

def tests_EpistasisLinearRegression_fit_sets_various_attributes():
    gpm = GenotypePhenotypeSimulation.from_length(2)
    model = EpistasisLinearRegression.from_gpm(gpm, order=2, model_type="local")
    model.fit()
    # Checks
    check1 = hasattr(model, "Xfit")
    check2 = hasattr(model, "coef_")
    check3 = hasattr(model, "epistasis")
    # Tests
    tools.assert_true(check1)
    tools.assert_true(check2)
    tools.assert_true(check3)

def tests_EpistasisLinearRegression_predict():
    gpm = GenotypePhenotypeSimulation.from_length(2)
    model = EpistasisLinearRegression.from_gpm(gpm, order=2, model_type="local")
    model.fit()
    check1 = model.predict()
    # Checks
    check1 = model.predict()
    # Tests
    np.testing.assert_almost_equal(check1, model.gpm.phenotypes)

def tests_EpistasisLinearRegression_predict_sets_Xpredict():
    gpm = GenotypePhenotypeSimulation.from_length(2)
    model = EpistasisLinearRegression.from_gpm(gpm, order=2, model_type="local")
    model.fit()
    y=model.predict()
    # Checks
    check1 = hasattr(model, "Xpredict")
    # Tests
    tools.assert_true(check1)

def tests_EpistasisLinearRegression_score():
    gpm = GenotypePhenotypeSimulation.from_length(2)
    model = EpistasisLinearRegression.from_gpm(gpm, order=2, model_type="local")
    model.fit()
    score = model.score()
    # Tests
    tools.assert_greater_equal(score, 0)
    tools.assert_less_equal(score, 1)

def tests_EpistasisLinearRegression_hypothesis():
    gpm = GenotypePhenotypeSimulation.from_length(2)
    model = EpistasisLinearRegression.from_gpm(gpm, order=2, model_type="local")
    model.fit()
    # Checks
    check1 = model.hypothesis(thetas=model.coef_)
    # Tests
    np.testing.assert_almost_equal(check1, model.gpm.phenotypes)
