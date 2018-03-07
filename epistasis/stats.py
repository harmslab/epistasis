__doc__ = """Submodule with useful statistics functions for epistasis model."""

# -----------------------------------------------------------------------
# Useful statistical metrics as methods
# -----------------------------------------------------------------------

import numpy as np
from scipy.stats import f
from scipy.stats import norm
import scipy

# -----------------------------------------------------------------------
# Correlation metrics
# -----------------------------------------------------------------------


def magnitude_vs_order(EpistasisMap, orders=None):
    """Calculate the average magnitude of epistasis for each order of epistasis.
    in a high-order epistasis model.
    """
    # If orders are not explicitly given
    if orders is None:
        orders = list(range(1, EpistasisMap.order + 1))
    else:
        if orders[-1] > EpistasisMap.order:
            raise Exception("""Epistatic orders given are larger than order of \
            epistasis in the EpistasisMap.""")
    magnitudes, std = [], []
    for order in orders:
        mapping = EpistasisMap.get_order(order)
        vals = abs(np.array(mapping.values))
        magnitudes.append(np.mean(vals))
        std.append(np.std(vals))
    return orders, magnitudes, std

# -----------------------------------------------------------------------
# Correlation metrics
# -----------------------------------------------------------------------


def gmean(x):
    """Calculate a geometric mean with zero and negative values.

    Following the gmean calculation from this paper:

    Habib, Elsayed AE. "Geometric mean for negative and zero values."
    International Journal of Research and Reviews in Applied Sciences 11
    (2012): 419-432.
    """
    x_neg = x[x < 0]
    x_pos = x[x > 0]
    x_zero = x[x == 0]

    n_neg = len(x_neg)
    n_pos = len(x_pos)
    n_zero = len(x_zero)
    N = len(x)

    gm_neg, gm_pos, gm_zero = 0, 0, 0
    if n_neg > 0:
        gm_neg = scipy.stats.mstats.gmean(abs(x_neg))
    if n_pos > 0:
        gm_pos = scipy.stats.mstats.gmean(x_pos)

    g1 = -1 * gm_neg * n_neg / N
    g2 = gm_pos * n_pos / N
    g3 = gm_zero * n_zero / N

    GM = g1 + g2 + g3
    return GM


def incremental_mean(old_mean, samples, M, N):
    """Calculate an incremental running mean.

    Parameters
    ----------
    old_mean : float or array
        current running mean(s) before adding samples
    samples : ndarray
        array containing the samples. Each column is a sample. Rows are
        independent values. Mean is taken across row.
    M : int
        number of samples in new chunk
    N : int
        number of previous samples in old mean
    """
    return ((N - M) * old_mean + samples.sum(axis=0)) / N


def incremental_var(old_mean, old_var, new_mean, samples, M, N):
    """Calculate an incremental variance.

    Parameters
    ----------
    old_mean : float or array
        current running mean(s) before adding samples
    old_var : float or array
        current running variance(s) before adding samples
    new_mean : float
        updated mean
    samples : ndarray
        array containing the samples. Each column is a sample. Rows are
        independent values. Mean is taken across row.
    M : int
        number of samples in new chunk
    N : int
        number of previous samples in old mean
    """
    return ((N - M) * old_var + np.array((samples - old_var) *
                                         (samples - new_mean)).sum(axis=0)) / N


def incremental_std(old_mean, old_std, new_mean, samples, M, N):
    """Calculate an incremental standard deviation.

    Parameters
    ----------
    old_mean : float or array
        current running mean(s) before adding samples
    samples : ndarray
        array containing the samples. Each column is a sample. Rows are
        independent values. Mean is taken across row.
    M : int
        number of samples in new chunk
    N : int
        number of previous samples in old mean
    """
    old_var = old_std**2
    return np.sqrt(incremental_var(old_mean, old_var, new_mean, samples, M, N))


def bootstrap(function, n_samples=10000, chunk_size=50, rtol=1e-2,
              *args, **kwargs):
    """Bootstrap a function until the mean and the std of the output has
    converged.

    Parameters
    ----------
    function : callable
        Function to repeat until convergence.
    n_samples : int
        max number of iterations to calculate for convergence.
    chunk_size : int
        number of samples to draw before recalculating the statistics for
        convergence.
    rtol : float
        relative tolerance for convergence over sampling

    *args : position arguments for the callable function that is resampled
    **kwargs : keyword arguments for the callable function that is sampled.

    Returns
    -------
    running_mean : numpy.array
        means of the samples
    running_std : numpy.array
        standard deviations of samples
    count : int
        number of samples needed to reach convergence test.
    """
    tests = (False, False)
    count = chunk_size
    running_mean = function(*args, **kwargs)
    try:
        size = len(running_mean)
        running_std = np.zeros(len(running_mean))
    except TypeError:
        size = 1
        running_std = 0
    while False in tests and count < n_samples:
        # If init_sample is array
        if size != 1:
            # Allocate a 2d array with proper size.
            new_samples = np.empty((chunk_size, size), dtype=float)
            # Populate the array with various samples
            for i in range(chunk_size):
                new_samples[i, :] = function(*args, **kwargs)
        # If it is a value
        else:
            # Allocate memory in 1d array
            new_samples = np.empty(chunk_size, float)
            for i in range(chunk_size):
                new_samples[i] = function(*args, **kwargs)
        # ########### Check for convergence ##############
        last_mean = np.copy(running_mean)
        last_std = np.copy(running_std)
        running_mean = incremental_mean(
            last_mean, new_samples, chunk_size, count)
        running_std = incremental_std(
            last_mean, last_std, running_mean, new_samples, chunk_size, count)
        # Check to see if all values pass the tolerance threshold
        check_mean = np.isclose(running_mean, last_mean, rtol=rtol)
        check_std = np.isclose(running_std, last_std, rtol=rtol)
        # Find any Falses
        test1 = np.all(check_mean)
        test2 = np.all(check_std)
        tests = (test1, test2)
        count += chunk_size
    return running_mean, running_std, count


def pearson(y_obs, y_pred):
    """ Calculate pearson coefficient between two variables.
    """
    x = y_obs
    y = y_pred

    xbar = np.mean(y_obs)
    ybar = np.mean(y_pred)

    terms = (x - xbar) * (y - ybar)

    numerator = sum(terms)

    # calculate denominator
    xdenom = sum((x - xbar)**2)
    ydenom = sum((y - ybar)**2)
    denominator = np.sqrt(xdenom) * np.sqrt(ydenom)

    return numerator / denominator


def rmsd(yobs, ypred):
    """Calculate the root mean squared deviation of an estimator."""
    ypred = np.array(ypred)
    yobs = np.array(yobs)
    return np.sqrt(np.sum((ypred - yobs)**2) / len(ypred))


def generalized_r2(y_obs, y_pred):
    """ Calculate the rquared between the observed and predicted y.
    See wikipedia definition of `coefficient of determination`.
    """
    # Mean fo the y observed
    y_obs_mean = np.mean(y_obs)
    # Total sum of the squares
    ss_total = sum((y_obs - y_obs_mean)**2)
    # Sum of squares of residuals
    ss_residuals = sum((y_obs - y_pred)**2)
    r_squared = 1 - (ss_residuals / ss_total)
    return r_squared


def explained_variance(y_obs, y_pred):
    """Returns the explained variance
    """
    # Mean fo the y observed
    y_obs_mean = np.mean(y_obs)
    # Total sum of the squares
    ss_total = sum((y_obs - y_obs_mean)**2)
    # Explained sum of squares
    ss_regression = sum((y_pred - y_obs_mean)**2)
    r_squared = (ss_regression / ss_total)
    return r_squared


def ss_residuals(y_obs, y_pred):
    """ calculate residuals """
    return sum((y_obs - y_pred)**2)


def chi_squared(y_obs, y_pred):
    """ Calculate the chi squared between observed and predicted y. """
    return sum((y_obs - y_pred)**2 / y_pred)

# -----------------------------------------------------------------------
# Model error statistics
# -----------------------------------------------------------------------


def false_positive_rate(y_obs, y_pred, upper_ci, lower_ci, sigmas=2):
    """ Calculate the false positive rate of predicted values. Finds all values
    that equal zero in the known array and calculates the number of false
    positives found in the predicted given the number of samples and sigmas.

    The defined bounds are:
        (number of sigmas) * errors / sqrt(number of samples)

    Parameters
    ----------
    known : array-like
        Known values for comparing false positives
    predicted : array-like
        Predicted values
    errors : array-like
        Standard error from model
    n_samples : int
        number of replicate samples
    sigma : int (default=2)
        How many standard errors away (2 == 0.05 false positive rate)

    Returns
    -------
    rate : float
        False positive rate in data
"""

    N = len(y_obs)
    # Check that known, predicted, and errors are the same size.
    if N != len(y_pred) or N != len(upper_ci):
        raise Exception("Input arrays must all be the same size")

    # Number of known-zeros:
    known_zeros = 0

    # Number of false positives:
    false_positives = 0

    # Scale confidence bounds to the number of samples and sigmas
    upper_bounds = sigmas * upper_ci
    lower_bounds = sigmas * lower_ci

    for i in range(N):
        # Check that known value is zero
        if y_obs[i] == 0.0:

            # Add count to known_zero
            known_zeros += 1

            # Calculate bounds with given number of sigmas.
            upper = y_pred[i] + upper_bounds[i]
            lower = y_pred[i] - lower_bounds[i]

            # Check false positive rate.
            if y_obs[i] > upper or y_obs[i] < lower:
                false_positives += 1

    # Calculate false positive rate
    rate = false_positives / float(known_zeros)

    return rate


def false_negative_rate(y_obs, y_pred, upper_ci, lower_ci, sigmas=2):
    """ Calculate the false negative rate of predicted values. Finds all values
    that equal zero in the known array and calculates the number of false
    negatives found in the predicted given the number of samples and sigmas.

    The defined bounds are:
        (number of sigmas) * errors / sqrt(number of samples)

    Parameters
    ----------
    known : array-like
        Known values for comparing false negatives
    predicted : array-like
        Predicted values
    errors : array-like
        Standard error from model
    n_samples : int
        number of replicate samples
    sigma : int (default=2)
        How many standard errors away (2 == 0.05 false negative rate)

    Returns
    -------
    rate : float
        False negative rate in data
    """

    N = len(y_obs)
    # Check that known, predicted, and errors are the same size.
    if N != len(y_pred) or N != len(upper_ci):
        raise Exception("Input arrays must all be the same size")

    # Number of known-zeros:
    known_nonzeros = 0

    # Number of false negatives:
    false_negatives = 0

    # Scale confidence bounds to the number of samples and sigmas
    upper_bounds = sigmas * upper_ci
    lower_bounds = sigmas * lower_ci

    for i in range(N):
        # Check that known value is zero
        if y_obs[i] != 0.0:

            # Add count to known_zero
            known_nonzeros += 1

            # Calculate bounds with given number of sigmas.
            upper = y_pred[i] + upper_bounds[i]
            lower = y_pred[i] - lower_bounds[i]

            # Check false negative rate.
            if lower < 0 < upper:
                false_negatives += 1

    # Calculate false positive rate
    rate = false_negatives / float(known_nonzeros)

    return rate
