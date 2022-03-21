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


def identify_peaks(data, error, line_width):
    # apply linear model to data
    # extract peak locations
    kernel_x = np.arange(-15, 16, 1)[::-1]
    kernel = gauss(kernel_x, 0.0, line_width)

    signal = np.convolve(kernel, data / error / error, mode='same')
    normalization = np.convolve(kernel * kernel, 1.0 / error / error, mode='same')

    metric = signal / normalization
    peaks, peak_properties = find_peaks(metric, height=50.0, distance=30.0)
    return peaks


def refine_peak_centers():
    # maybe maximize a gaussian weight filter with a variable line widths and center?
    pass


def correlate_peaks(peaks, linear_model, lines, match_threshold):
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
