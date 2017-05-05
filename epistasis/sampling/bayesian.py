
"""
A Bayesian, high-order, linear epistasis model.

"""
import numpy as _np
import emcee as _emcee

class BayesianSampler(object):

    def __init__(self, model):
        self.model

    def lnlike(theta, x, y, yerr):
        m, b, lnf = theta
        model = m * x + b
        inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
        return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))
