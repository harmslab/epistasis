import networkx as nx
import matplotlib.pyplot as plt
from scipy.misc import comb

def neighbors(length, label):
    """ 
    Pass sequence as string and return neighbors. 
    If mapping, report neighbors as their index in mapping.
    """
    neighbors = list()
    # Generate higher order neighbors
    set_label = set(label)
    sites = set(range(1,length+1))
    diff = list(sites - set_label)
    for d in diff:
        neighbors.append(",".join([str(i) for i in sorted(label + [d])]))
    
    # Generate lower order neighbors
    if len(label) > 1:        
        for i in range(len(label)):
            label_copy = list(label)
            label_copy.pop(i)
            neighbors.append(",".join([str(j) for j in label_copy]))
    
    return neighbors

def form_edges(length, interaction_labels):
    """ Build edges between all sequences based on single mutations. """
    edges = list()
    interaction_labels.remove([0])
    for s in interaction_labels:
        nbs = neighbors(length, s)
        edges = edges + [(",".join([str(i) for i in s]), nb) for nb in nbs] 
    return edges
    

def interactions_to_position(length, interaction_keys):
    """ Trick to create node positions for genotype-phenotype map. """
    # Use Pascal's triangle structure (comb) to position nodes.
    pascals = [(comb(length,k)-1)/2.0 for k in range(0,length+1)]
    positions = dict()
    for s in interaction_keys:
        x = len(s.split(","))
        y = pascals[x]
        pascals[x] = pascals[x]-1
        positions[s] = (y,-x)
    return positions
    

class EpistasisGraph(object):
    
    def __init__(self, em):
        self.em = em
        self.labels = list(self.em.interaction_labels)
        self.edges = form_edges(self.em.length, self.labels)
        self.G = nx.Graph(self.edges)
        
        self.colorbar = None        
        self.attributes = {}
        self.clean_graph()
    
    def clean_graph(self):
        """ Return networkx graph to default colors and sizes. """
        self.attributes = {
                            "pos":interactions_to_position(self.em.length,self.G.nodes()),
                            "with_labels":False,
                            "node_size": 300,
                            "node_color": 'w',
                            "cmap": None,
                            "width":0.3
                            }


    def color_mapping(self):
        """ Add colors to nodes. """
        self.attributes["cmap"] = plt.cm.bwr
        self.attributes["node_color"] = list()
        for node in self.G.nodes():
            self.attributes["node_color"].append(self.em.interaction_mapping[node])


    def create_colorbar(self):
        vmin = min(self.attributes["node_color"])
        vmax = max(self.attributes["node_color"])
        sm = plt.cm.ScalarMappable(cmap=self.attributes["cmap"], norm=plt.normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        self.colorbar = plt.colorbar(sm)
        self.colorbar.ax.set_ylabel('Interaction Value', fontsize=20)
        
    def node_size(self):
        """ Size of nodes reflect value of each node. """
        self.attributes["node_size"] = list()
        for node in self.G.nodes():
            self.attributes["node_size"].append(400*abs(self.em.interaction_mapping[node]))       
    
    def draw_network(self, colorbar=False, figsize=[15,8]):
        """ Return a figure of sequence space. """        
        # Create a matplotlib Axes object
        fig, ax = plt.subplots(1,1, figsize=figsize)
        fig.dpi = 300
        ax.set_axis_off()
        
        # Draw the networkx graph
        nx.draw_networkx(self.G, **self.attributes)
        
        # Add labels offset above the node
        label_pos = dict()
        pos = self.attributes["pos"]
        for p in pos:
            label_pos[p] = (pos[p][0], pos[p][1]+.2)
        nx.draw_networkx_labels(self.G, label_pos)
        
        # Add a colorbar
        if colorbar is True:
            self.create_colorbar()
            
