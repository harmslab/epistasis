__doc__ = """
Submodule with classes for comparing two epistasis models. 
"""

import numpy as np

def overlap(items1, items2):
    """ Get overlapping items. """
    overlaps = list()
    for i1 in items1:
        for i2 in items2:
            if i1 == i2:
                overlaps.append(i1)
    return overlaps

def rmsd(val1, val2):
    """ Calculate the rmsd between two values. """
    if len(val1) != len(val2):
        raise Exception("Two values must the the same length.")
    
    # Convert lists into numpy array.
    if type(val1) is list:
        val1 = np.array(val1)
        val2 = np.array(val2)
        
    return np.sqrt(sum((abs(val1) - abs(val2))**2)/len(val1))

class ModelTypeError(Exception):
    """ Raise this exception if models being compared are different types. """

class ModelComparison(object):
    
    def __init__(self, model1, model2):
        if type(model1) != type(model2):
            raise ModelTypeError("""The two models being compared are not the same type. 
                                    (i.e. Both GlobalEpistasModel? or LocalEpistasisModel?)""")
        self.model1 = model1
        self.model2 = model2
        
    @property
    def genotype_overlap(self):
        """ Get the genotypes that overlap. """
        return overlap(self.model1.genotypes1, self.model2.genotypes2)
    
    @property
    def interaction_overlap(self):
        """ Get interactions that overlap. """
        return overlap(self.model1.Interactions.genotypes, self.model2.Interactions.genotypes)
    
    @property
    def interaction_values(self):
        """ Get all interaction values that are the same between two models."""
        overlap = self.interaction_overlap
        model1map = self.model1.genotype2value
        model2map = self.model2.genotype2value
        values = list()
        keys = list()
        for o in overlap:
            keys.append(o)
            values.append((model1map[o], model2map[o]))
        return keys, values
    
    @property
    def rmsd_interactions(self):
        """ Get the rmsd between two items from the model. """
        keys,vals = self.Interactions.values
        val1 = np.array(vals)[:,0]
        val2 = np.array(vals)[:,1]
        return rmsd(val1,val2)
        