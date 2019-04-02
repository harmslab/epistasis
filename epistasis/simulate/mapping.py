from functools import wraps
from ..mapping import EpistasisMap
from numpy import random

class DistributionException(Exception):
    """"""

class SimulatedEpistasisMap(EpistasisMap):
    """Just like an epistasis map, but with extra methods
    for setting epistatic coefficients
    """
    def __init__(self, gpm, df=None, sites=None, values=None, stdeviations=None):
        super().__init__(df=df, sites=sites, values=values, stdeviations=stdeviations)
        self._gpm = gpm

    @property
    def avail_distributions(self):
        return random.__all__

    def set_order_from_distribution(self, orders, dist="normal", **kwargs):
        """Sets epistatic coefficients to values drawn from a statistical distribution.

        Distributions are found in SciPy's `random` module. Kwargs are passed
        directly to these methods
        """
        # Get distribution
        try: 
            method = getattr(random, dist)
        except AttributeError:
            raise DistributionException("Distribution now found. Check the `avail_distribution` "
                                        "attribute for available distributions.")

        idx = self.data.orders.isin(orders)
        self.data.loc[idx, "values"] = method(
            size=sum(idx),
            **kwargs
        )
        self._gpm.build()

    @wraps(EpistasisMap.set_values)
    def set_values(self, values, filter=None):
        super().set_values(values, filter=filter)
        self._gpm.build()