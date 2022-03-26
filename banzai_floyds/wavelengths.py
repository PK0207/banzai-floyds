import numpy as np
from numpy.polynomial.legendre import Legendre
from banzai_floyds.matched_filter import matched_filter_metric
from scipy.signal import find_peaks


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
    Get best fit first-order wavelength solution

    Parameters
    ----------
    data: array of 1D raw spectrum extraction
    error: array of uncertainties
            Same shapes as the input data array
    lines: table containing 'wavelength' and 'strength' for each standard line
    dispersion: float
        Guess of Angstroms per pixel
    line_width: average line width in angstroms
    offset_range: array
        Range of values to search for the offset in the linear wavelength solution

    Returns
    -------
    linear model function that takes an array of pixels and outputs wavelengths
    """
    # Step the model spectrum metric through each of the offsets and find the peak
    slope = dispersion * (len(data) // 2)
    metrics = [matched_filter_metric((offset, slope), data, error, wavelength_model_weights, None, None,
                                     np.arange(data.size), lines, line_width) for offset in offset_range]
    best_fit_offset = offset_range[np.argmax(metrics)]
    return Legendre((best_fit_offset, slope), domain=(0, len(data) - 1))


def identify_peaks(data, error, line_width, line_sep):
    """
        Detect peaks in spectrum extraction

        Parameters
        ----------
        data: array of 1D raw spectrum extraction
        error: array of uncertainties
                Same shapes as the input data array
        line_width: average line width in angstroms
        line_sep: minimum separation distance before lines are determined to be unique

        Returns
        -------
        array containing the location of detected peaks
        """
    # extract peak locations
    kernel_x = np.arange(-15, 16, 1)[::-1]
    kernel = gauss(kernel_x, 0.0, line_width)

    signal = np.convolve(kernel, data / error / error, mode='same')
    normalization = np.convolve(kernel * kernel, 1.0 / error / error, mode='same')

    metric = signal / normalization
    peaks, peak_properties = find_peaks(metric, height=50.0, distance=line_sep)
    return peaks


def refine_peak_centers():
    # maybe maximize a gaussian weight filter with a variable line widths and center?
    pass


def correlate_peaks(peaks, linear_model, lines, match_threshold):
    """
        Find the standard line peaks associated with the detected peaks in a raw 1D arc extraction

        Parameters
        ----------
        peaks: array containing the pixel location of detected peaks
        linear_model: 1st order fit function for the wavelength solution
        lines: table containing 'wavelength' and 'strength' for each standard line
        match_threshold: maximum separation for a pair of peaks to be considered a match.

        Returns
        -------
        list of standard line peak wavelengths matching detected peaks
        """
    guessed_wavelengths = linear_model(peaks)
    corresponding_lines = []
    # correlate detected peaks to known wavelengths
    for peak in guessed_wavelengths:
        corresponding_line = lines['wavelength'][np.argmin(np.abs(peak - lines['wavelength']))]
        if np.abs(corresponding_line - peak) >= match_threshold:
            corresponding_line = None
        corresponding_lines.append(corresponding_line)
    return corresponding_lines


def estimate_distortion(peaks, corresponding_wavelengths, domain, order=4):
    return Legendre.fit(deg=order, x=peaks, y=corresponding_wavelengths, domain=domain)
