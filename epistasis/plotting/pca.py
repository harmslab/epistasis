import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact

# -----------------------------
# Plotting PCA
# -----------------------------

def principal_components(model, with_components=True, dimensions=3, figsize=[6,6], ac="r"):
    """
    Parameters
    ----------
    model : EpistasisPCA object
        PCA model of epistasis to plot
    figsize : array like
        Figure size
    ac : str
        Color for arrows representing principal components
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if dimensions > 3:
        raise Exception(""" Cannot plot greater than 3 dimensions! """)

    # Plot the principal components?
    if with_components:

        x,y,z  = 0,0,0

        U1 = model.components[0][0] * 0.03
        V1 = model.components[0][1] * 0.03
        W1 = model.components[0][2] * 0.03
        U2 = model.components[1][0] * 0.03
        V2 = model.components[1][1] * 0.03
        W2 = model.components[1][2] * 0.03
        U3 = model.components[2][0] * 0.03
        V3 = model.components[2][1] * 0.03
        W3 = model.components[2][2] * 0.03

        # PCA
        a = Arrow3D([-U1,U1], [-V1,V1], [-W1,W1], mutation_scale=20, lw=2, arrowstyle="->", color="r")
        b = Arrow3D([-U2,U2], [-V2,V2], [-W2,W2], mutation_scale=20, lw=2, arrowstyle="->", color="r")
        c = Arrow3D([-U3,U3], [-V3,V3], [-W3,W3], mutation_scale=20, lw=2, arrowstyle="->", color="r")

        ax.add_artist(a)
        ax.add_artist(b)
        ax.add_artist(c)

    # Print transformed phenotypes?
    for i in model.X_new:
        ax.scatter(i[0],i[1],i[2], s=100)

    mapping = model.get_map("indices", "X_new")
    model.Graph.build_graph()

    for edge in model.Graph.edges():
        ax.plot((mapping[edge[0]][0], mapping[edge[1]][0]),
                (mapping[edge[0]][1], mapping[edge[1]][1]),
                (mapping[edge[0]][2], mapping[edge[1]][2]),
                linewidth=2,
               c='grey')

    @interact
    def angle(angle=(0,360,10), elevation=(-100,100,10)):
        ax.view_init(azim=angle, elev=elevation)
        return fig
