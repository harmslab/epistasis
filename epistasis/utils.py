# -------------------------------------------------------
# Miscellaneous Python functions (hacks) for random task
# -------------------------------------------------------
import itertools as it
import numpy as np
from scipy.misc import comb
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


def list_binary(length):
    """ List all binary strings with given length. """
    return np.array(sort(["".join(seq) for seq in it.product("01", repeat=length)]))
    

def epistatic_order_indices(length, order):
    """ Return  the indices of interactions with the given order. 
        
        Args:
        ----
        length: int
            length of the sequences
        order: int
            order of interactions to return
    """
    start = int(sum([comb(length, i) for i in range(order)]))
    stop = int(start + comb(length, order))
    return start, stop


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
    mutation_map = dict(zip(mutations, range(n_mut)))
    
    # Enumerate mutations flipping combinations
    combinations = np.array([list(j) for i in range(1,n_mut+1) for j in it.combinations(rev_mutations, i)])
    # Initialize empty arrays 
    sequence_space = np.empty(size, dtype="<U" + str(len(wildtype)))
    binaries = np.empty(size, dtype="<U" + str(n_mut))
    # Population first element with wildtypes
    sequence_space[0] = wildtype
    binaries[0] = binary_wt
    # Iterate through mutations combinations and build binary representations
    counter = 1
    for c in combinations:
        sequence = list(wildtype)
        b = list(binary_wt)
        for el in c:
            sequence[el] = mutant[el]   # Sequence version of mutant
            b[mutation_map[el]] = '1'            # Binary version of mutant
        sequence_space[counter] = "".join(sequence)
        binaries[counter] = "".join(b)
        counter += 1
     
    if binary:
        return sequence_space, binaries
    else:
        return sequence_spaces
