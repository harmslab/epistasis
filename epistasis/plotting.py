import numpy as np
import matplotlib.pyplot as plt
    
# ---------------------------------------------------
# Epistasis Graphing
# ---------------------------------------------------

def epistasis_bar(epistasis_map, sigmas=0, title="Epistatic interactions", string_labels=False, ax=None, color='b', figsize=[6,4]):
    """ Plot the interactions sorted by their order. 
    
    Parameters:
    ----------
    title: str
        The title for the plot.
    sigmas: 
        Number of sigmas to represent the errorbars. If 0, no error bars will be included.
    """
    em = epistasis_map
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    else:
        fig = ax.get_figure()
    
    y = em.Interactions.values
    if string_labels is True:
        xtick = em.Interactions.genotypes
    else:
        xtick = em.Interactions.keys
        xlabel = "Interaction Indices"
    
    # plot error if sigmas are given.
    if sigmas == 0:
        ax.bar(range(len(y)), y, 0.9, alpha=0.4, align="center", color=color) #, **kwargs)
    else:
        yerr = em.Interactions.errors
        ax.bar(range(len(y)), y, 0.9, yerr=sigmas*yerr, alpha=0.4, align="center", color=color) #,**kwargs)
    
    # vertically label each interaction by their index
    plt.xticks(range(len(y)), np.array(xtick), rotation="vertical", family='monospace',fontsize=7)
    ax.set_ylabel("Interaction Value", fontsize=14) 
    try:
        ax.set_xlabel(xlabel, fontsize=14)
    except:
        pass
    ax.set_title(title, fontsize=12)
    ax.axis([-.5, len(y)-.5, -max(abs(y)), max(abs(y))])
    ax.hlines(0,0,len(y), linestyles="dashed")
    return fig, ax    

def epistasis_barh(epistasis_map, sigmas=0, title="Epistatic interactions", 
                    string_labels=False, ax=None, color='b', figsize=[6,4], partition=False):
    """ Plot the interactions sorted by their order. 
    
    Parameters:
    ----------
    title: str
        The title for the plot.
    sigmas: 
        Number of sigmas to represent the errorbars. If 0, no error bars will be included.
    """
    em = epistasis_map
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    else:
        fig = ax.get_figure()
    
    x = em.Interactions.values
    if string_labels is True:
        ytick = em.Interactions.genotypes
        ylabel = "Interactions"
    else:
        ytick = em.Interactions.keys
        ylabel = "Interaction Indices"
    
    # plot error if sigmas are given.
    if sigmas == 0:
        ax.barh(-np.arange(len(x)), x, 0.9, alpha=0.4, align="center", color=color) #, **kwargs)
    else:
        xerr = em.Interactions.errors
        ax.barh(-np.arange(len(x)), x, 0.9, xerr=sigmas*xerr, alpha=0.4, align="center", color=color) #,**kwargs)
    
    # vertically label each interaction by their index
    plt.yticks(-np.arange(len(x)), np.array(ylabels), rotation="horizontal", family='monospace', fontsize=7)
    ax.set_xlabel("Interaction Value", fontsize=16, fontname='monospace') 
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=20)
    ax.axis([-max(abs(x)), max(abs(x)), -len(x)+.5, .5])
    ax.vlines(0,0,-len(x), linestyles="dashed")
    return fig, ax    

def epistasis_bar_charts(em, sigmas=0, x_axis=[-1.5,1.5]):
    """ Plot each order of epistasis from model as barcharts subplots.
    
    Parameters:
    ----------
    em: EpistasisModel object
        Must be a complete, fitted epistasis map
    order: int
        Highest order of epistasis to plot
    """
    length = em.length
    order = em.order
    fig, ax = plt.subplots(order, 1, figsize=[5,2*order])

    for o in range(1, order+1):
        interactions, errors = em.get_order(o, errors=sigmas)
        labels = list(interactions.keys())
        values = list(interactions.values())
        n_terms = len(values)
        index = np.arange(n_terms)
        bar_width = .8
        opacity = 0.5
        rects1 = ax[o-1].barh(index, values, bar_width,
                         alpha=opacity,
                         color='b',
                         xerr=sigmas*np.array(list(errors.values())),
                        align="center",
                        ecolor='r')
        ticks = ax[o-1].set_yticklabels(labels, rotation="horizontal", fontsize=8)
        ax[o-1].set_yticks(index)
        
        axis_dim = list(x_axis) + [-1,len(index)]
        ax[o-1].axis(axis_dim) 
        ax[o-1].vlines(0,-1,len(index), linestyles=u'solid')
        if o != order:
            ax[o-1].get_xaxis().set_ticks([])
    ax[0].set_title("Order of Interaction: " +str(order))

def ensemble_bar(ensemble, title="Ensemble Epistasis"):
    """ Return a bar chart of ensemble interactions from an ensemble model calculation. """
    fig, ax = plt.subplots(1,1, figsize=[12,6])
    averages, deviations = ensemble.ensemble_averages()
    xlabels = list()
    y = list()
    yerr = list()
    for key in averages:
        y.append(averages[key])
        yerr.append(deviations[key])
        xlabels.append(key)
        
    ax.bar(range(len(y)), y, 0.9, yerr=yerr, alpha=0.4, align="center") #,**kwargs)
    
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

def correlation(learned, known, title="Known vs. Learned", figsize=[6,6]):
    """ Create a plot showing the learned data vs. known data. """
    
    fig, ax = plt.subplots(1,1, dpi=300, figsize=figsize)
    
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
    
def residuals(learned, known, title="Residual Plot", figsize=[6,4]):
    """ Generate a residual plot. """
    fig, ax = plt.subplots(1,1, dpi=300, figsize=figsize)
    
    ax.stem(known, (learned-known), 'b-', markerfmt='.')
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("True")
    ax.set_ylabel("Residuals")
    
    return fig

