import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def corr(y_pred, y_obs, fig=None, ax=None, figsize=None):
    """Make a correlation plot.
    """
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(111)
    # Make correlation plot
    z = np.linspace(min(y_pred), max(y_pred),2)
    ax.plot(y_pred, y_obs, '.')
    ax.plot(z,z, '--', color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axis("equal")
    return fig, ax

def resid(y_pred, y_obs, fig=None, ax=None, figsize=None):
    """Make a residual plot between y_obs and y_pred (i.e. y_obs - y_pred).
    """
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(111)
    residuals = y_obs - y_pred
    # Make residual plot
    ax.plot(y_pred, residuals, '.')
    ax.hlines(0,min(y_pred), max(y_pred), linestyle="--", color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax

def rhist(y_pred, y_obs, fig=None, ax=None, figsize=None):
    """Make a horizontal histogram for residuals between y_obs and y_pred.
    """
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot(111)
    residuals = y_obs - y_pred
    # Make histogram plot
    ax.hist(residuals, bins=10, orientation="horizontal")
    ax.hlines(0,min(y_pred), max(y_pred), linestyle="--", color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax

def corr_resid(y_pred, y_obs, figsize=None):
    """Make a correlation and residuals plot using matplotlib.

    Parameters
    ----------
    y_pred : array
        predicted values (will be placed on x axis)
    y_obs : array
        observed values (will be placed on y axis).

    Returns
    -------
    fig : Figure object
        Figure object
    ax_c : Axes object
        subplot with correlation plot drawn
    ax_r : Axes object
        subplot with residual plot drawn
    """
    # Initialize the grid.
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[9,1])
    ax_c = fig.add_subplot(gs[0, :])
    ax_r = fig.add_subplot(gs[1 ,:], sharex=ax_c)

    # build subplots
    corr(y_pred, y_obs, fig=fig, ax=ax_c)
    resid(y_pred, y_obs, fig=fig, ax=ax_r)
    return fig, ax_c, ax_r

def corr_resid_rhist(y_pred, y_obs, figsize=(3,4)):
    """Make a correlation, residuals, histogram plot using matplotlib.

    Parameters
    ----------
    y_pred : array
        predicted values (will be placed on x axis)
    y_obs : array
        observed values (will be placed on y axis).

    Returns
    -------
    fig : Figure object
        Figure object
    ax_c : Axes object
        subplot
    ax_r : Axes object
        subplot with residual plot drawn
    ax_h : Axes object
        subplot with histogram plot drawn
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2,2, height_ratios=[4,1], width_ratios=[4,1])
    # Construct grid.
    ax_c = fig.add_subplot(gs[0:1, 0:1])
    ax_r = fig.add_subplot(gs[1: , 0:1], sharex=ax_c)
    ax_h = fig.add_subplot(gs[1: , 1: ], sharey=ax_r)

    corr(y_pred, y_obs, fig=fig, ax=ax_c)
    resid(y_pred, y_obs, fig=fig, ax=ax_r)
    rhist(y_pred, y_obs, fig=fig, ax=ax_h)
    return fig, ax_c, ax_r, ax_h
