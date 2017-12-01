
import unittest
import pytest
from gpmap import GenotypePhenotypeMap
from ..mixed import EpistasisMixedRegression
from .. import EpistasisLinearRegression, EpistasisLogisticRegression
import warnings

# Ignore fitting warnings
warnings.simplefilter("ignore", RuntimeWarning)


@pytest.fixture
def gpm():
    """Create a genotype-phenotype map"""
    wildtype = "000"
    genotypes = ["000", "001", "010", "100", "011", "101", "110", "111"]
    phenotypes = [2.5838167335880149,
                  2.4803514336043708,
                  2.2205925336075762,
                  2.1864673462520905,
                  1.5622922695718136,
                  1.8972733199455831,
                  1.3324426002143119,
                  1.7367637632162392]
    stdeviations = 0.01
    return GenotypePhenotypeMap(wildtype, genotypes, phenotypes,
                                stdeviations=stdeviations)


class TestEpistasisMixedRegression(object):

    # Set some initial parameters for this model
    order = 3
    threshold = 1.5622922695718136

    def test_init(self, gpm):
        classifier = EpistasisLogisticRegression(order=1,
                                                 threshold=self.threshold)
        epi = EpistasisLinearRegression(order=self.order)
        model = EpistasisMixedRegression(classifier, epi)
        assert hasattr(model, "Model")
        assert hasattr(model, "Classifier")

    def test_add_gpm(self, gpm):
        classifier = EpistasisLogisticRegression(order=1,
                                                 threshold=self.threshold)
        epi = EpistasisLinearRegression(order=self.order)
        model = EpistasisMixedRegression(classifier, epi)
        model.add_gpm(gpm)
        assert hasattr(model, "gpm")
        assert hasattr(model.Model, "gpm")
        assert hasattr(model.Classifier, "gpm")

    def test_fit(self, gpm):
        classifier = EpistasisLogisticRegression(order=1,
                                                 threshold=self.threshold)
        epi = EpistasisLinearRegression(order=self.order)
        model = EpistasisMixedRegression(classifier, epi)
        model.add_gpm(gpm)
        model.fit()
        assert hasattr(model, "Model")

    def test_predict(self, gpm):
        classifier = EpistasisLogisticRegression(order=1,
                                                 threshold=self.threshold)
        epi = EpistasisLinearRegression(order=self.order)
        model = EpistasisMixedRegression(classifier, epi)
        model.add_gpm(gpm)
        model.fit()
        predicted = model.predict(X="complete")
        assert "predict" in model.Model.Xbuilt
        assert len(predicted) == gpm.n

    def test_thetas(self, gpm):
        classifier = EpistasisLogisticRegression(order=1,
                                                 threshold=self.threshold)
        epi = EpistasisLinearRegression(order=self.order)
        model = EpistasisMixedRegression(classifier, epi)
        model.add_gpm(gpm)
        model.fit()

        coefs = model.thetas
        # Tests
        assert len(coefs) == 12

    def test_hypothesis(self, gpm):

        classifier = EpistasisLogisticRegression(order=1,
                                                 threshold=self.threshold)
        epi = EpistasisLinearRegression(order=self.order)
        model = EpistasisMixedRegression(classifier, epi)
        model.add_gpm(gpm)
        model.fit()

        predictions = model.hypothesis()
        # Need more thorough tests
        assert len(predictions) == gpm.n

    def test_lnlikelihood(self, gpm):
        classifier = EpistasisLogisticRegression(order=1,
                                                 threshold=self.threshold)
        epi = EpistasisLinearRegression(order=self.order)
        model = EpistasisMixedRegression(classifier, epi)
        model.add_gpm(gpm)
        model.fit()

        # Calculate lnlikelihood
        lnlike = model.lnlikelihood()
        assert lnlike.dtype == float
