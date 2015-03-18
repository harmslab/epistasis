import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Epistasis Graphing
# ---------------------------------------------------

def epistasis_bar_charts(em, length, order):
    """ Generate stacked subplots, showing barcharts of interactions for each order
        of epistasis. 
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
        
