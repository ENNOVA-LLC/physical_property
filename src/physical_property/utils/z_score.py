""" `utils.z_score`

This module contains a function to calculate the z-score of an array using a uniform filter.
"""
import numpy as np
from scipy import ndimage

def z_score(arr: np.ndarray, size: int):
    """
    Calculates the z-score of an array using a uniform filter.

    Parameters
    ----------
    arr : np.ndarray
        The input array to calculate the z-score for.
    size : int
        The size of the filter window used in the uniform filter.

    Returns
    -------
    np.ndarray
        The z-score array.
    """
    c1 = ndimage.uniform_filter1d(arr, size=size)
    c2 = ndimage.uniform_filter1d(arr * arr, size=size)
    var = c2 - c1 * c1
    var = np.where(var < 1.e-12, 0., var)
    sigma = np.sqrt(var)  # standard deviation
    mu = c1  # mean value

    # z-score
    return np.divide(arr - mu, sigma, where=(sigma > 0), out=np.zeros_like(arr), dtype=np.float64)
