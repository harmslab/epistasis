import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = "Arial"

def prettify(ax,
    axis_border=0.05,
    x_spine_limit=None,
    y_spine_limit=None,

    ):
    """ A simple wrapper to make matplotlib figures prettier."""
    extra_limit_frac = 0.05
    spine_widths = 1.35
    line_widths = 1.5
    errorbars = False

    # Only prettify the first time.
    if hasattr(ax, "prettify") is False:
        ax.prettify = True

        # Get current axis limits
        xlimits = list(ax.get_xlim())
        ylimits = list(ax.get_ylim())
        xticks = list(ax.get_xticks())
        yticks = list(ax.get_yticks())

        # Extend the graph by 5 percent on all sides
        xextra = extra_limit_frac*(xlimits[1] - xlimits[0])
        yextra = extra_limit_frac*(ylimits[1] - ylimits[0])

        # set ticks and tick labels
        ax.set_xlim(xlimits[0] - xextra, xlimits[1] + xextra)
        ax.set_ylim(ylimits[0] - yextra, ylimits[1] + yextra)

        # Remove right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Set the bounds for visible axes
        ax.spines['bottom'].set_bounds(xlimits[0], xlimits[1])
        ax.spines['left'].set_bounds(ylimits[0], ylimits[1])

        # Thicken the spines
        ax.spines['bottom'].set_linewidth(spine_widths)
        ax.spines['left'].set_linewidth(spine_widths)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # Make ticks face outward and thicken them
        ax.tick_params(direction='out', width=spine_widths)

        if xticks[-1] > xlimits[1]:
            xticks = xticks[:-1]

        if yticks[-1] > ylimits[1]:
            yticks = yticks[:-1]

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
