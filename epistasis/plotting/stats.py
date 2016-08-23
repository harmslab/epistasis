import matplotlib.pyplot as plt
import numpy as np
from seqspace.plotting import mpl_missing
from scipy.stats import f

class FDistributionPlotting(object):

    @mpl_missing # Don't use this plotting object in outside classes if mpl is not installed
    def __init__(self, FDistribution):
        """Plotting a distribution. """
        self._dist = FDistribution

    def pdf(self, percent_start=0.0001, percent_end=0.9999, figsize=(6,4), **kwargs):
        """ Plot the distribution. """
        #Build distribution
        x = np.linspace(self._dist.ppf(percent_start),
                        self._dist.ppf(percent_end), 1000)

        y = self._dist.pdf(x)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, **kwargs)
        ax.set_title("Probability density function")
        ax.set_xlabel("F-statistic")
        ax.set_ylabel("Probability density")
        return fig, ax

    def cdf(self, percent_start=0.0001, percent_end=0.9999, figsize=(6,4), **kwargs):
        #Build distribution
        x = np.linspace(self._dist.ppf(percent_start),
                        self._dist.ppf(percent_end), 1000)

        y = self._dist.cdf(x)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, **kwargs)
        ax.set_title("Cumulative distribution function")
        ax.set_xlabel("F-statistic")
        ax.set_ylabel("Cumulative probability")

        return fig, ax

# ---------------------------------------------------
# Epistasis Graphing
# ---------------------------------------------------


def magnitude_vs_order(model, keep_sign=False,
        marker="o",
        color="b",
        linestyle="None",
        xlabel="Order",
        ylabel="Magnitude",
        title="",
        figsize=(6,4),
        errorbars=True,
        **kwargs):
    """
        Generate a plot of magnitude versus order.

    """
    orders = range(1, model.length+1)
    magnitudes = []
    errors = []
    for i in orders:
        coeffs = np.array(list(model.Interactions.get_order(i).values()))

        # Do we care about signs?
        if keep_sign is False:
            coeffs = abs(coeffs)

        # Add magnitudes
        magnitudes.append(np.mean(coeffs))
        errors.append(np.std(coeffs))

    # Initialize the figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Create plot
    if errorbars:
        ax.errorbar(orders, magnitudes, yerr=errors, marker=marker, color=color, linestyle=linestyle, **kwargs)

    else:
        ax.plot(orders, magnitudes, marker=marker, color=color, linestyle=linestyle, **kwargs)

    # Steal the max bound, cause matplotlib does a great job of picking good axis
    ylimits = max(list(ax.get_ylim()))
    # Make y limits symmetric
    ax.axis([orders[0], orders[-1], -ylimits, ylimits])




    # Add a line at zero if keeping sign
    if keep_sign:
        ax.hlines(0, orders[0], orders[-1], linestyle="--")

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
