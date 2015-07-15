import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm as scipy_norm
    
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


def bar_with_xbox(model,
                  order_colors=("red","orange","green","purple","DeepSkyBlue","yellow","pink"),
                  significance="bon",significance_cutoff=0.05,
                  sigmas=2,y_scalar=1.5,y_axis_name="interaction",
                  figsize=(8,10),height_ratio=12,
                  star_cutoffs=(0.05,0.01,0.001),star_spacer=0.0075):
    """
    Create a barplot with the values from model, drawing the x-axis as a grid of
    boxes indicating the coordinate of the epistatic parameter. Should automatically
    generate an almost publication-quality figure.

    input:
        model: epistasis model
        order_colors: list/tuple of colors for each order (rgb,html string-like)
        significance: how to treat signifiance.  should be:
                      "bon" -> Bonferroni corrected p-values (default)
                      "p" -> raw p-values
                      None -> ignore significance
        significance_cutoff: value above which to consider a term significant
        sigmas: number of sigmas to show for each error bar
        y_scalar: how much to scale the y-axis above and beyond y-max
        y_axis_name: what to put on the y-axis of the barplot

        figsize: tuple of figure width,height
        height_ratio: how much to scale barplot relative to xbox
        star_cutoffs: signifiance cutoffs for star stack.  should go from highest
                      p to lowest p (least to most significant)
        star_spacer: constant that scales how closely stacked stars are from one
                     another
        
    output:
        pretty-graph objects fig and ax_array (ax_array has two entries for top
        and bottom panels)
    """
    # Grab the fit values and labels, tossing 0th-order term
    bar_y = model.Interactions.values[1:]
    labels = model.Interactions.labels[1:]
    
    # Figure out the length of the x-axis and the highest epistasis observed
    num_terms = len(labels)
    highest_order = max([len(l) for l in labels])
    
    # Figure out how to color each order
    if order_colors == None:
        order_colors = ["gray" for i in range(highest_order+1)]
    else:
        if len(order_colors) < highest_order:
            err = "order_colors has too few entries (at least {:d} needed)\n".format(highest_order)
            raise ValueError(err)
        
        # Stick gray in the 0 position for insignificant values
        order_colors = list(order_colors)
        order_colors.insert(0,"gray")
  
    # ---------------------- #  
    # Deal with significance #
    # ---------------------- #
        
    z_score = -model.Interactions.values[1:]/model.Interactions.errors[0][1:]

    # straight p-values
    if significance == "p":
        p_values = 2*scipy_norm.cdf(-abs(z_score))

    # bonferroni corrected p-values
    elif significance == "bon":
        p_values = 2*scipy_norm.cdf(-abs(z_score))*len(model.Interactions.values)

    # ignore p-values and color everything
    elif signifiance == None:
        p_values = [0 for i in range(len(labels))]
        significance_cutoff = 1.0
   
    # or die 
    else:
        err = "signifiance argument {:s} not recognized\n".format(significance)
        raise ValueError(err)   
    
    # Create color array based on significance
    color_array = np.zeros((len(labels)),dtype=int)
    for i, l in enumerate(labels):        
        if p_values[i] < significance_cutoff:
            color_array[i] = len(l) - 1
        else:
            color_array[i] = 0
        
    # ---------------- #
    # Create the plots #
    # ---------------- #
    
    # Make a color map
    cmap = matplotlib.colors.ListedColormap(colors=order_colors)
    cmap.set_bad(color='w', alpha=0) # set the 'bad' values (nan) to be white and transparent
    bounds = range(len(order_colors))
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    # Create a plot with an upper and lower panel, sharing the x-axis
    fig = plt.figure(figsize=figsize)
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[height_ratio, 1])
    ax_array = [plt.subplot(gs[0])]
    ax_array.append(plt.subplot(gs[1],sharex=ax_array[0]))
    
    # ------------------ #
    # Create the barplot #
    # ------------------ #

    # set up bar colors
    colors_for_bar = np.array([matplotlib.colors.colorConverter.to_rgba(order_colors[i]) for i in color_array])

    # Plot error if sigmas are given.
    if sigmas == 0:
        ax_array[0].bar(range(len(bar_y)), bar_y, 0.9,color=colors_for_bar)
    else:
        yerr = model.Interactions.errors[0][1:]
        ax_array[0].bar(range(len(bar_y)), bar_y, 0.9, yerr=sigmas*yerr,color=colors_for_bar,
                        error_kw={"ecolor":"black"})
    
    # Label barplot y-axis
    ax_array[0].set_ylabel(y_axis_name, fontsize=14) 
    
    # Set barplot y-scale
    ymin = -y_scalar*max(abs(bar_y))
    ymax =  y_scalar*max(abs(bar_y))
    
    # Make axes pretty pretty
    ax_array[0].axis([-1, len(bar_y) + 1, ymin, ymax])
    ax_array[0].set_frame_on(False) #axis("off")
    ax_array[0].get_xaxis().set_visible(False)
    ax_array[0].get_yaxis().tick_left()
    ax_array[0].get_yaxis().set_tick_params(direction='out')
    ax_array[0].add_artist(matplotlib.lines.Line2D((-1,-1), (ax_array[0].get_yticks()[1], ax_array[0].get_yticks()[-2]), color='black', linewidth=1))
 
    # add vertical lines between order breaks
    previous_order = 1
    for i in range(len(labels)):
        if len(labels[i]) != previous_order:
            ax_array[0].add_artist(matplotlib.lines.Line2D((i,i),
                                                           (ymin,ymax),
                                                           color="gray",
                                                           linestyle="--"))
            previous_order = len(labels[i])

    # ------------------------- #
    # Create significance stars #
    # ------------------------- #

    min_offset = star_spacer*(ymax-ymin)
    for i in range(len(p_values)):

        star_counter = 0
        for j in range(len(star_cutoffs)):
            if p_values[i] < star_cutoffs[j]:
                star_counter += 1
            else:
                break

        for j in range(star_counter):
            ax_array[0].text(x=(i+0.5),y=ymin+(j*min_offset),s="*") 

    # --------------------------- #
    # Create the box-array x-axis #
    # --------------------------- #
    
    # make an empty data set
    data = np.ones((highest_order,num_terms),dtype=int)*np.nan
 
    # Make entries corresponding to each coordinate 1
    for i, l in enumerate(labels):
        for j in l:
            data[(j-1),i] = color_array[i]
    
    # draw the grid
    for i in range(num_terms + 1):
        ax_array[1].add_artist(matplotlib.lines.Line2D((i,i),
                                                       (0,highest_order),
                                                       color="black"))
                                                
    for i in range(highest_order+1):
        ax_array[1].add_artist(matplotlib.lines.Line2D((0,num_terms),
                                                       (i,i),
                                                       color="black"))
        
    # draw the boxes
    ax_array[1].imshow(data, interpolation='nearest',cmap=cmap,norm=norm,
                       extent=[0, num_terms, 0, highest_order], zorder=0)
    
    # turn off the axis labels
    ax_array[1].set_frame_on(False) 
    ax_array[1].axis("off")
    ax_array[1].set_xticklabels([])
    ax_array[1].set_yticklabels([])
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    
    # Draw the final figure
    fig.tight_layout()
    
    return fig, ax_array
      

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

