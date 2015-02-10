import numpy as np
import networkx as nx
from ..geno_pheno_map.gpm_network import GenoPhenoGraph

# ---------------------------------------------------
# NetworkX graphing
# ---------------------------------------------------

def





class EpistasisGraph(GenoPhenoGraph):
    
    def __init__(self, geno_pheno_map, regression_model):
        self.gpm = geno_pheno_map
        self.model = regression_model
        
    def node_mapping(self):
        
        node_to_pheno = dict()
        node_to_geno = dict()
        
        # Dict with interaction term to interaction value
        node_to_interaction = self.model.get_interactions()
        
        for key in node_to_interactions:
            
            node_to_pheno[key] = 
        