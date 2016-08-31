import scipy

def reverse_power_transform(y, lmbda, A, B):
    """reverse"""
    gmean = scipy.stats.mstats.gmean(model.statistics.linear()+A)
    return (gmean**(lmbda-1)*lmbda*(y - B) + 1)**(1/lmbda) - A
