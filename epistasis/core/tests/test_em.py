import unittest
import numpy as np
from em import EpistasisMap

assertArraysEqual = np.testing.assert_array_equal

class EpistasisMapTestCase(unittest.TestCase):
    
    def setUp(self):
        self.em = EpistasisMap()
        
    def tearDown(self):
        self.em = None
        
    def test_genotypes(self):
        genotypes = np.array(['00','01', '10','11'])
        phenotypes = np.random.rand(len(genotypes))
        self.em.genotypes = genotypes
        self.em.phenotypes = phenotypes
        assertArraysEqual(genotypes, self.em.genotypes)
        assertArraysEqual(phenotypes, self.em.phenotypes)

if __name__ == '__main__':
    unittest.main()