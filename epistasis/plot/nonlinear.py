import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def corr(padd, pobs, perr=None, ax=None):
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
    ax.plot(padd, pobs, 'o', color='k')
    ax.plot(z,z, '--', color='gray')
    
    # Set spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set axis
    ax.axis('square')
    
    return fig, ax


def model(model):
    
    padd = model.Additive.predict("fit")
    pobs = model.gpm.phenotypes
    
    fig, ax = corr(padd, pobs)
    
    # Built a model line
    xmodel = np.linspace(min(padd), max(padd), 1000)
    ymodel = model.function(xmodel, *model.parameters.get_params())
    
    # Plot model
    ax.plot(xmodel, ymodel, '-', linewidth=2, color='r')
    
    return fig, ax
    
    
