import matplotlib.pyplot as plt
import numpy as np

def epistatic_coeffs(self, model1, model2):
    """
    """
    fig, ax = plt.subplots(figsize=(2,2))

    orders = range(2, linear.length+1)
    colors = ("red","orange","green","purple","DeepSkyBlue","yellow","pink")

    for o in orders:
        mapping = linear.epistasis.getorder[o]
        z2 = mapping.values
        z = mean[mapping.indices]
        ax.plot(z2, z, '.', markersize=6, color=colors[o-1])

    r2 = pearson(linear.epistasis.values[10:], mean[10:])**2

    ax.annotate(s="r$^{2}$ = " + str(round(r2,3)), xy=(0.0001,-0.00035))
    ax.axis("equal")
    ax.axis([-.00055, .00055, -0.00055, 0.00055])
    #plt.axis("equal")
    size = ax.axis()
    t = np.linspace(size[0],size[1], 10)
    ax.plot(t,t, ":", color="gray", zorder=0)
    #ax.hlines(0, -1,1, linewidth=.5)
    #ax.vlines(0, -1,1, linewidth=.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_position('center')
    ax.spines["left"].set_position('center')
    #np.r
    ax.set_xticks([-.0005,-.00025, .00025, .0005])
    ax.set_yticks([-.0005,-.00025, .00025, .0005])
    ax.set_xticklabels([-0.0005,'','',0.0005])
    ax.set_yticklabels([-0.0005,'','',0.0005])
    ax.tick_params(direction="inout")
    return fig, ax
