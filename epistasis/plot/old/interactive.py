import numpy as np
from ipywidgets import interactive as _interactive
from ipywidgets import fixed as _fixed
from .epistasis import epistasis as _epistasis_plot
from collections import OrderedDict

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
    options = OrderedDict(
        save=False,
        fname="figure.svg",
        format="svg",
        y_axis_name="interaction",
        xgrid=True,
        figwidth=(1,20, .5),
        figheight=(1,20, .5),
        y_scalar=(0,5,.1),
        height_ratio=(0,10,1),
        star_spacer=(0.000,0.1,0.001),
        significance=["bon", "p", None],
        significance_cutoff=_fixed(0.05),
        sigmas=(0,5,.5),
        ecolor="black",
        capthick=(0,2,.1),
        capsize=(0,2,.1),
        elinewidth=(0,5,.1),
        log_space=False,
        log_transform=False,
    )
    types = dict([(key, type(val)) for key, val in options.items()])
    for key, value in kwargs.items():
        typed = types[key]
        if typed == np.ufunc:
            typed_val = value
        elif options[key] == None:
            typed_val = value
        else:
            typed_val = types[key](value)
        options[key] = typed_val

    w = _interactive(_epistasis_plot,
        betas=_fixed(betas),
        labels=_fixed(labels),
        errors=_fixed(errors),
        **options
    )

    return w
