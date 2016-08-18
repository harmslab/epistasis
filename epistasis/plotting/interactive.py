import numpy as np
from ipywidgets import interactive as _interactive
from ipywidgets import fixed as _fixed
from .epistasis import epistasis as _epistasis_plot

def savewrapper(func, *args, **kwargs):
    """Wrapper to save figures."""
    def inner(fname, format, save=False, *args, **kwarg):
        print(**kwargs)
        print("hi")
        fig, ax = func(*args, **kwargs)
        if save:
            fig.savefig(fname, format=format, bbox_height="tight")
        return fig, ax
    return inner

def epistasis(betas, labels, errors=[], **kwargs):
    """Create a widget for interactive epistasis plots.
    """
    options = {
        "save" : False,
        "fname" : "figure.svg",
        "format" : "svg",
        "y_scalar" : (0,5,.1),
        "log_transform" : False,
        "height_ratio" : 12,
        "star_spacer" : 0.0075,
        "significance" : ["bon", "p", None],
        "significance_cutoff" : 0.05,
        "sigmas" : (0,5,.5),
        "log_space" : False,
        "y_axis_name" : "interaction",
        "xgrid" : True,
        "figwidth" : 5,
        "figheight" : 3,
        "bar_borders" : True,
        "ecolor" : "black",
        "capthick": (0,2,.1),
        "capsize" : (0,2,.1),
        "elinewidth" : (0,5,.1),
    }

    w = _interactive(_epistasis_plot,
        betas=_fixed(betas),
        labels=_fixed(labels),
        errors=_fixed(errors),
        **options,
    )

    return w
