import numpy as np
from sklearn.metrics import mean_squared_error

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