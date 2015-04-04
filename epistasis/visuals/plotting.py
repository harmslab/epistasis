import numpy as np
import matplotlib.pyplot as plt

def epistasis_bar(epistasis_map, sigmas=0, title="Epistatic interactions"):
    """ Plot the interactions sorted by their order. 
    
    Parameters:
    ----------
    title: str
        The title for the plot.
    sigmas: 
        Number of sigmas to represent the errorbars. If 0, no error bars will be included.
    """
    em = epistasis_map
    fig, ax = plt.subplots(1,1, figsize=[12,6])
    y = em.interaction_values
    xlabels = em.interaction_keys
    
    # plot error if sigmas are given.
    if sigmas == 0:
        ax.bar(range(len(y)), y, 0.9, alpha=0.4, align="center") #, **kwargs)
    else:
        yerr = em.interaction_errors
        ax.bar(range(len(y)), y, 0.9, yerr=sigmas*yerr, alpha=0.4, align="center") #,**kwargs)
    
    # vertically label each interaction by their index
    plt.xticks(range(len(y)), np.array(xlabels), rotation="vertical")
    ax.set_xlabel("Interaction term", fontsize=16)
    ax.set_ylabel("Interaction Value", fontsize=16) 
    ax.set_title(title, fontsize=20)
    ax.axis("tight")
    ax.hlines(0,0,len(y), linestyles="dashed")
    return fig, ax    


# -----------------------------
# Useful plots for analyzing data 
# from regression data.
# -----------------------------

def correlation(learned, known, title="Known vs. Learned"):
    """ Create a plot showing the learned data vs. known data. """
    
    fig, ax = plt.subplots(1,1, dpi=300)
    
    ax.plot(known, learned, '.b')
    ax.hold(True)
    x = np.linspace(min(known), max(known), 1000)
    ax.plot(x,x, '-r', linewidth=1)
    ax.set_xlabel("Known", fontsize = 14)
    ax.set_ylabel("Learned", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    return fig
    
def residuals(learned, known, title="Residual Plot"):
    """ Generate a residual plot. """
    fig, ax = plt.subplots(1,1, dpi=300)
    
    ax.stem(known, (learned-known), 'b-', markerfmt='.')
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("True")
    ax.set_ylabel("Residuals")
    
    return fig
    
# ---------------------------------------------------
# Epistasis Graphing
# ---------------------------------------------------

def epistasis_bar_charts(em, length, order):
    """ Generate stacked subplots, showing barcharts of interactions for each order
        of epistasis. 
        
        BROKEN
    """
    fig, ax = plt.subplots(length, 1, figsize=[5,5*order])

    for order in range(1, length+1):
        interactions = em.nth_order(order)
        error = em.nth_error(order)
        labels = interactions.keys()
        values = interactions.values()
        n_terms = len(values)
        index = np.arange(n_terms)
        bar_width = .9
        opacity = 0.4
        rects1 = ax[order-1].bar(index, values, bar_width,
                         alpha=opacity,
                         color='b',
                         yerr=error.values())
        ticks = ax[order-1].set_xticklabels(labels, rotation="vertical")
        ax[order-1].set_xticks(index+.5)
        
