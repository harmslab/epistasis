import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def corr(padd, pobs, perr=None, ax=None, color='k'):
    """Make a correlation plot.
    """
    # Get
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Make correlation plot
    z = np.linspace(min(padd), max(padd), 2)

    # Add data to plot
    ax.scatter(padd, pobs, marker='o', c=color, cmap="coolwarm", vmin=0, vmax=1)
    ax.plot(z,z, '--', color='gray')
    
    # Set spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set axis
    ax.axis('square')
    return fig, ax

def model(model):

    # Get Padd vs Pobs datapoints and classes
    padd = model.Model.Additive.predict(X="obs")
    pprob = model.Classifier.predict_proba(X="obs")
    pobs = model.gpm.phenotypes

    # Build a color array for the two classes
    #color = np.empty(len(pclass), dtype=str)
    #color[pclass==0] = "r"
    #color[pclass==1] = "k"

    # Construct the correlation plot between Pobs and Padd
    fig, ax = corr(padd, pobs, color=pprob[:,1])

    # Built a model line
    xmodel = np.linspace(min(padd), max(padd), 1000)
    ymodel = model.Model.function(xmodel, *model.Model.parameters.get_params())

    # Plot model
    ax.plot(xmodel, ymodel, '-', linewidth=4, color='k', alpha=0.5)
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    # Plot the "dead" region on the Pobs axis.
    ybox1 = [model.threshold, model.threshold]
    ybox2 = [ylim[0], ylim[0]]
    xbox = [xlim[0], xlim[1]]
    ax.fill_between(xbox, ybox1, ybox2, alpha=0.1, color='C0')
    return fig, ax
