import numpy as np


def gauss(x, mu, sigma):
    """
    return a normal distribution
    Parameters
    ----------
    x: array of x values
    mu: center/mean/median of normal distribution
    sigma: standard deviation of normal distribution
    Returns
    -------
    array of y values corresponding to x values in given normal distribution
    """
    return 1 / np.sqrt(2.0 * np.pi) / sigma * np.exp(-0.5 * (x - mu) * (x - mu) / sigma / sigma)


def fwhm_to_sigma(fwhm):
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def sigma_to_fwhm(sigma):
    return sigma * (2.0 * np.sqrt(2.0 * np.log(2.0)))
