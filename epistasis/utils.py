# -------------------------------------------------------
# Miscellaneous Python functions (hacks) for random task
# -------------------------------------------------------
import itertools as it
import numpy as np
from sklearn.metrics import mean_squared_error

def hamming_distance(s1, s2):
    """ Return the Hamming distance between equal-length sequences """
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

def find_differences(s1, s2):
    """ Return the index of differences between two sequences."""
    indices = list()
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            indices.append(i)
    return indices

def enumerate_space(wildtype, mutant, binary=True):
    """ Generate binary genotype space between two sequences. 
    
        Args:
        ----
        wildtype: str
            Wildtype sequence as starting reference point.
        mutant: str
            Mutant sequence. 
            
        Returns:
        -------
        sequence_space: list
            List of all sequence combinations between the two sequences. 
        binary_representation: list (optional)
            In
            
        Example:
        -------
        if wildtype == 'AAA' and mutant == 'TTT':
            sequence space =    ['AAA','AAV','AVA','VAA','AVV','VAV','VVA','VVV']
    """
    
    # Check that wildtype and mutant are the same length
    if len(wildtype) != len(mutant):
        raise IndexError("ancestor_sequence and derived sequence must be the same length.")
    
    # Count mutations and keep indices
    mutations = find_differences(wildtype, mutant)
    n_mut = len(mutations)
    binary_wt = "".zfill(n_mut)
    size = 2**n_mut
    rev_mutations = [mutations[i] for i in range(n_mut-1, -1, -1)]
    mutation_map = dict(zip(range(n_mut), mutations))
    
    # Enumerate mutations flipping combinations
    combinations = np.array([list(j) for i in range(1,n_mut+1) for j in it.combinations(rev_mutations, i)])
    # Initialize empty arrays 
    sequence_space = np.empty(size, dtype="<U" + str(n_mut))
    binaries = np.empty(size, dtype="<U" + str(n_mut))
    # Population first element with wildtypes
    sequence_space[0] = wildtype
    binaries[0] = binary_wt
    # Iterate through mutations combinations and build binary representations
    counter = 1
    for c in combinations:
        sequence = list(wildtype)
        binary = list(binary_wt)
        for el in c:
            sequence[el] = mutant[el]   # Sequence version of mutant
            binary[el] = '1'            # Binary version of mutant
        sequence_space[counter] = "".join(sequence)
        binaries[counter] = "".join(binary)
        counter += 1
     
    if binary:
        return sequence_space, binaries
    else:
        return sequence_spaces

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
    