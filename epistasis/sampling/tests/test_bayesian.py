#
#
# import pytest
#
# from gpmap import GenotypePhenotypeMap
# from epistasis.models import EpistasisLinearRegression
# from ..bayesian import BayesianSampler
#
# @pytest.fixture
# def model():
#     """Create a genotype-phenotype map"""
#     wildtype = "000"
#     genotypes =  ["000", "001", "010", "100", "011", "101", "110", "111"]
#     phenotypes = [  0.0,   0.1,   0.5,   0.4,   0.2,   0.8,   0.5,   1.0]
#     stdeviations = 0.01
#     gpm = GenotypePhenotypeMap(wildtype, genotypes, phenotypes, stdeviations=stdeviations)
#     model = EpistasisLinearRegression.read_gpm(gpm, order=2).fit()
#     return model
#
# class TestBayesianSampler(object):
#
#     def test_init(self, model):
#         sampler = BayesianSampler(model)
#
#         assert hasattr(sampler, "model")
#         assert hasattr(sampler, "lnlikelihood")
#         assert hasattr(sampler, "ml_thetas")
#         assert hasattr(sampler, "lnprior")
#         assert hasattr(sampler, "ndim")
#         assert hasattr(sampler, "nwalkers")
#         assert hasattr(sampler, "sampler")
#         assert hasattr(sampler, "last_run")
#
#     def test_lnprob(self, model):
#         sampler = BayesianSampler(model)
#         output = sampler.lnprob(model.thetas, model.lnlikelihood)
#
#         assert output.dtype == float
#
#     def test_get_initial_walkers(self, model):
#         sampler = BayesianSampler(model)
#         walkers = sampler.get_initial_walkers()
#
#         assert walkers.shape == (sampler.nwalkers, sampler.ndim)
#
#     def test_sample(self, model):
#         sampler = BayesianSampler(model)
#         sampler.sample(5)
#
#         assert hasattr(sampler.sampler, "chain")
#         assert sampler.samples.shape == (5*sampler.nwalkers, sampler.ndim)
#         assert sampler.last_run != None
#         assert type(sampler.last_run) == tuple
#
#     def test_predict(self, model):
#         sampler = BayesianSampler(model)
#         sampler.sample(5)
#         p = sampler.predict()
#
#         assert p.shape == (5*sampler.nwalkers, len(model.gpm.complete_genotypes))
