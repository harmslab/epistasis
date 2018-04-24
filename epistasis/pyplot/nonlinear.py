import functools
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_scale(
    model=None,
    yadd=None,
    yobs=None,
    yerr=None,
    function=None,
    cmap=None,
    color=None,
    s=50,
    ax=None,
    function_line=True,
    line_color='r',
    **kwargs):
    """Plot a Y_obs vs. Y_add showing the nonlinear scale in a genotype-phenotype
    map.

    Parameters
    ----------
    model : (default=None)
        Epistasis model.

    yadd : array (default=None)
        x-axis data. The additive model phenotypes.

    yobs : array (default=None)
        y-axis data. The observed phenotypes.

    yerr : array (default=None)
        y-axis error. Error in observed phenotypes.

    function : callable
        Nonlinear function.

    cmap : str
        Colormap name to map onto scatter points.

    color : str, array,
        color of phenotypes.

    s : int
        size of scatter points.

    ax : Axes
        Axes object to plot points on.

    function_line : bool
        If true, plots nonlinear function on top of points.

    line_color : matplotlib color.
        color of function line.


    Returns
    -------
    ax : matplotlib.Axes
        Axes object with plot.
    """
    if model is not None:
        params = model.parameters

        yobs = model.gpm.phenotypes
        yadd = model.Additive.predict()

        xx = np.linspace(min(yadd), max(yadd),20)
        yy = model.minimizer.predict(xx)

    elif yobs is None or yadd is None:
        raise Exception("If not model is given, pobs and padd must be set.")

    elif len(yobs) != len(yadd):
        raise Exception("pobs and padd must be same length.")

    # Set colormap and normalize colors.
    if cmap is not None:
        # Get colormap
        cmap_ = getattr(mpl.cm, cmap)
        cmap = cmap_(yobs)

        # Normalize the colors
        norm = mpl.colors.Normalize(vmin=min(yobs), vmax=max(yobs))
        color = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_).to_rgba(yobs)

    # If no colormap, set color to black
    else:
        color = 'k'

    # Prepare plot
    if ax is None:
        fig, ax = plt.subplots()

    # Plot data.
    ax.scatter(yadd, yobs, color=color, s=s)

    # Plot errorbars if given
    if yerr is not None:
        ax.errorbar(yadd, yobs, yerr=yerr, fmt="none", ecolor='gray', zorder=-1)

    # Plot model
    if function_line:
        ax.plot(xx, yy, color=line_color, linewidth=2)
    return ax


@functools.wraps(plot_scale)
def plot_power_transform(
    model=None,
    yobs=None,
    yadd=None,
    function_line=True,
    function=None,
    line_color='r',
    *args,
    **kwargs):

    # Get data
    if model is not None:
        params = model.parameters

        yobs = model.gpm.phenotypes
        yadd = model.Additive.predict()
        xdata = yadd

        xx = np.linspace(min(yadd), max(yadd),20)
        yy = model.function(xx, **params, data=xdata)

    elif yobs is None or yadd is None or function is None:
        raise Exception("If not model is given, pobs and padd must be set.")

    elif len(yobs) != len(yadd):
        raise Exception("pobs and padd must be same length.")

    ax = plot_scale(
        model=model,
        yobs=yobs,
        yadd=yadd,
        function_line=False,
        function=function,
        *args, **kwargs,
    )

    # Plot model
    if function_line:
        ax.plot(xx, yy, color=line_color, linewidth=2)

    return ax
