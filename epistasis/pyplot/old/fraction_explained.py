import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib import patches
import warnings

def fraction_explained(fraction_explained, color_vector=None, num_bins=1000,lw=0.25):
    """
    Plot a square "pie" chart where each box is colored according to how often it
    occurs in the data set.

    Parameters
    ----------
    """
    # Normalize fx_vector so it adds up to 1.0
    internal_fx_vector = np.array(np.copy(fraction_explained))
    if np.sum(internal_fx_vector) != 1.0:
        warnings.warn("fx_vector does not add up to 1")
        internal_fx_vector = internal_fx_vector/np.sum(internal_fx_vector)

    # Create a color vector or grab the one off the command line
    if color_vector is None:
        # Prepare an cycle of colors
        order = len(fraction_explained)
        prop_cycle = plt.rcParams['axes.prop_cycle']
        color_vector = prop_cycle.by_key()['color']
        color_scalar = int(order / len(color_vector))  + 1
        color_vector *= color_scalar
    else:
        if len(fx_vector) > len(color_vector):
            err = "len(color_vector) must be >= len(fx_vector)\n"
            raise ValueError(err)

    # Discretize the input vector with appropriately scaled
    side_length = np.int(np.round(np.sqrt(num_bins),0))
    num_bins = side_length*side_length
    fx_vector_int = np.cumsum(np.round(internal_fx_vector*num_bins,0)).astype(int)

    # Generate the plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # path codes for drawing the boxes
    box_codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY]

    # Go through each bin and color appropriately
    current_index = 0
    total_counter = 0
    for i in range(side_length-1,-1,-1):
        for j in range(side_length):
            if total_counter >= fx_vector_int[current_index]:
                current_index += 1
            # kind of a hack.  last entry sometimes has round error and doesn't get given a color.
            # use last entry to fill in.
            if current_index >= len(fraction_explained):
                current_index -= 1
            # Draw box
            verts = [(j,i),(j+1,i),(j+1,i+1),(j,i+1),(j, i),]
            path = Path(verts, box_codes)
            patch = patches.PathPatch(path, facecolor=color_vector[current_index], lw=lw)
            ax.add_patch(patch)
            total_counter += 1

    # Clean up plot
    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-1,side_length+1)
    ax.set_ylim(-1,side_length+1)
    return fig, ax
