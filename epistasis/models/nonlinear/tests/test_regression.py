# Externel imports
import numpy as np
from nose import tools
from gpmap.simulate import GenotypePhenotypeSimulation

# Module to test
from ..regression import *

class Toolkit(object):

    def __init__(self):
        self.gpm = GenotypePhenotypeSimulation.from_length(2)

    @staticmethod
    def function(x, A, B):
        return A*x + B

    @staticmethod
    def reverse(y, A, B):
        return (y - B) / A

def test_EpistasisNonlinearRegression_initialization():
    toolkit = Toolkit()
    model = EpistasisNonlinearRegression.from_gpm(toolkit.gpm,
        function=toolkit.function,
        reverse=toolkit.reverse,
        order=2,
        model_type="local")
    # Checks
    check1 = hasattr(model, 'parameters')
    check2 = hasattr(model, 'function')
    check3 = hasattr(model, 'reverse')
    # Tests
    tools.assert_true(check1)
    tools.assert_true(check2)
    tools.assert_true(check3)

def test_EpistasisNonlinearRegression_init_sets_parameter_object():
    toolkit = Toolkit()
    model = EpistasisNonlinearRegression.from_gpm(toolkit.gpm,
        function=toolkit.function,
        reverse=toolkit.reverse,
        order=2,
        model_type="local")
    parameters = model.parameters
    # Checks
    check1 = hasattr(parameters, 'A')
    check2 = hasattr(parameters, 'B')
    # Tests
    tools.assert_true(check1)
    tools.assert_true(check2)

def test_EpistasisNonlinearRegression_fit_sets_Xfit():
    toolkit = Toolkit()
    model = EpistasisNonlinearRegression.from_gpm(toolkit.gpm,
        function=toolkit.function,
        reverse=toolkit.reverse,
        order=2,
        model_type="local")
    model.fit(A=1,B=0)
    # Checks
    check1 = hasattr(model, 'Xfit')
    # Tests
    tools.assert_true(check1)

def test_EpistasisNonlinearRegression_predict():
    toolkit = Toolkit()
    model = EpistasisNonlinearRegression.from_gpm(toolkit.gpm,
        function=toolkit.function,
        reverse=toolkit.reverse,
        order=2,
        model_type="local")
    model.fit(A=1,B=0)
    y = model.predict()
    # Tests
    np.testing.assert_almost_equal(y, model.gpm.phenotypes)

def test_EpistasisNonlinearRegression_thetas():
    toolkit = Toolkit()
    model = EpistasisNonlinearRegression.from_gpm(toolkit.gpm,
        function=toolkit.function,
        reverse=toolkit.reverse,
        order=2,
        model_type="local")
    model.fit(A=1,B=0)
    coefs = model.thetas
    # Tests
    tools.assert_equals(len(coefs), 6)

def test_EpistasisNonlinearRegression_hypothesis():
    toolkit = Toolkit()
    model = EpistasisNonlinearRegression.from_gpm(toolkit.gpm,
        function=toolkit.function,
        reverse=toolkit.reverse,
        order=2,
        model_type="local")
    model.fit(A=1,B=0)
    predictions = model.hypothesis()
    # Tests
    np.testing.assert_almost_equal(predictions, model.gpm.phenotypes)
