import numpy as np
from numpy.polynomial.legendre import Legendre
from banzai_floyds.matched_filter import matched_filter_metric


def gauss(x, mu, sigma):
    return 1 / np.sqrt(2.0 * np.pi) / sigma * np.exp(-0.5 * (x - mu) * (x - mu) / sigma / sigma)


def wavelength_model_weights(theta, x, lines, line_width):
    wavelength_model = Legendre(theta, domain=(np.min(x), np.max(x)))
    wavelengths = wavelength_model(x)
    weights = np.zeros(x.shape)
    for line in lines:
        weights += line['strength'] * gauss(wavelengths, line['wavelength'], line_width)
    return weights


def linear_wavelength_solution(data, error, lines, dispersion, line_width, offset_range):
    """

    Parameters
    ----------
    data
    error
    lines
    dispersion: float
        Guess of Angstroms per pixel
    line_width
    offset_range: array
        Range of values to search for the offset in the linear wavelength solution

    Returns
    -------

    """
    # Step the model spectrum metric through each of the offsets and find the peak
    slope = dispersion * (len(data) // 2)
    metrics = [matched_filter_metric((offset, slope), data, error, wavelength_model_weights, None, None,
                                     np.arange(data.size), lines, line_width) for offset in offset_range]
    best_fit_offset = offset_range[np.argmax(metrics)]
    return Legendre((best_fit_offset, slope), domain=(0, len(data) - 1))


def extract_peaks(data, error, linear_model):
    # apply linear model to data
    # extract peak locations
    #
    pass


def identify_peaks(data, error, linear_model, lines):
    # extract peaks
    # correlate detected peaks to known wavelengths
    # create model for detected lines
    pass
