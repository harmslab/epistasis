# -------------------------------------------------------
# Miscellaneous Python functions (hacks) for random task
# -------------------------------------------------------
import itertools as it
import numpy as np
from sklearn.metrics import mean_squared_error

def hamming_distance(s1, s2):
    """ Return the Hamming distance between equal-length sequences """
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

def generate_binary_space(wildtype, mutant):
    """ Generate binary genotype space between two sequences (that should differ at all sites) """
    if len(wildtype) != len(mutant):
        raise IndexError("ancestor_sequence and derived sequence must be the same length.")

    binaries = sorted(["".join(list(s)) for s in it.product('01', repeat=len(wildtype))])
    sequence_space = list()
    for b in binaries:
        binary = list(b)
        sequence = list()
        for i in range(len(wildtype)):
            if b[i] == '0':
                sequence.append(wildtype[i])
            else:
                sequence.append(mutant[i])
        sequence_space.append(''.join(sequence))
    return sequence_space

def interaction_error_vs_order(learned, known, order):
    """ Take learned and known interaction dicts. """
    # Initializing a dictionary to hold order
    order_dict = dict()
    for i in range(order):
        order_dict[i+1] = list()
        
    for k in learned.keys():
        int_order = len(k.split(","))
        if k not in known:
            mse = np.sqrt((learned[k])**2)
        else:
            mse = np.sqrt((learned[k]-known[k])**2)
        order_dict[int_order].append(mse)
    
    mse = np.empty(order, dtype=float)
    std = np.empty(order, dtype=float)
    for i in range(order):
        mse[i] = np.mean(order_dict[i+1])
        std[i] = np.std(order_dict[i+1])
        
    return mse, std, range(1,order+1)
    
def error_window(mse, std, interaction_labels):
    """ Makes an array for plotting interaction uncertainty window. """
    err_window = np.empty(len(interaction_labels), dtype=float)
    std_window = np.empty(len(interaction_labels), dtype=float)
    for i in range(len(interaction_labels)):
        order = len(interaction_labels[i])
        err_window[i] = mse[order-1]
        std_window[i] = std[order-1]
        
    return err_window, std_window
    