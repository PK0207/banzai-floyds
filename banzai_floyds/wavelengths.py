import numpy as np
from numpy.polynomial.legendre import Legendre
from banzai.stages import Stage
from banzai_floyds.matched_filter import matched_filter_metric
from scipy.signal import find_peaks
from banzai_floyds.matched_filter import maximize_match_filter


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
        line_width: average line width (fwhm) in angstroms
        line_sep: minimum separation distance before lines are determined to be unique

        Returns
        -------
        array containing the location of detected peaks
        """
    # extract peak locations
    # Assume +- 3 sigma for the kernel width
    kernel_half_width = int(3 * line_width / 2.355)
    kernel_x = np.arange(-kernel_half_width, kernel_half_width + 1, 1)[::-1]
    kernel = gauss(kernel_x, 0.0, line_width)

    signal = np.convolve(kernel, data / error / error, mode='same')
    normalization = np.convolve(kernel * kernel, 1.0 / error / error, mode='same')

    metric = signal / normalization
    peaks, peak_properties = find_peaks(metric, height=50.0, distance=line_sep)
    return peaks


def centroiding_weights(theta, x):
    center, line_width = theta
    sigma = line_width / (2 * np.sqrt(2 * np.log(2)))
    return gauss(x, center, sigma)


def refine_peak_centers(data, error, peaks, line_width):
    """
        Find a precise center and width based on a gaussian fit to data

        Parameters
        ----------
        data: array of 1D raw spectrum extraction
        error: array of uncertainties
                Same shapes as the input data array
        peaks: array containing the pixel location of detected peaks
        line_width: average line width in angstroms

        Returns
        -------
        list of fit parameters for each peak:
            Gaussian fit parameters: peak center, standard deviation, scale
    """
    line_sigma = line_width / 2.355
    half_fit_window = int(3 * line_sigma)
    centers = []
    for peak in peaks:
        window = slice(peak - half_fit_window, peak + half_fit_window + 1, 1)
        data_window = data[window]
        error_window = error[window]
        x = np.arange(-half_fit_window, half_fit_window + 1, dtype=float)
        best_fit_center, best_fit_line_width = maximize_match_filter((0, line_width), data_window, error_window,
                                                                     centroiding_weights, x)
        centers.append(best_fit_center + peak)
    return centers


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
    """

    :param peaks: list of detected peaks in Pixel coordinates
    :param corresponding_wavelengths: list of peaks in physical units
    :param domain: tuple with minimum and maximum x value for given order
    :param order: int order of fitting polynomial
    :return:
    """
    return Legendre.fit(deg=order, x=peaks, y=corresponding_wavelengths, domain=domain)


def full_wavelength_solution_weights(theta, coordinates, lines):
    """
    Produce a 2d model of arc fluxes given a line list and a wavelength solution polynomial, a tilt, and a line width

    Parameters
    ----------
    theta: tuple: tilt, line_width, *polynomial_coefficients
    coordinates: tuple of 2d arrays x, y. x and y are the coordinates of the data array for the model
    lines: astropy table of the lines in the line list with wavelength (in angstroms) and strength

    Returns
    -------
    model array: 2d array with the match filter weights given the wavelength solution model
    """
    tilt, line_width, *polynomial_coefficients = theta
    x, y = coordinates
    tilted_x = x + np.tan(np.deg2rad(tilt)) * y
    wavelength_polynomial = Legendre(polynomial_coefficients, domain=(np.min(x), np.max(x)))
    model_wavelengths = wavelength_polynomial(tilted_x)
    model = np.zeros_like(model_wavelengths)
    line_sigma = line_width / (2 * np.sqrt(2 * np.log(2)))
    for line in lines:
        # in principle we should set the resolution to be a constant, i.e. delta lambda / lambda, not the overall width
        model += line['strength'] * gauss(model_wavelengths, line['wavelength'], line_sigma)
    return model


def full_wavelength_solution(data, error, x, y, initial_polynomial_coefficients, initial_tilt, initial_line_width,
                             lines):
    """
    Use a match filter to estimate the best fit 2-d wavelength solution

    Parameters
    ----------
    data: 2-d array with data to be fit
    error: 2-d array error, same shape as data
    x: 2-d array, x-coordinates of the data, same, shape as data
    y: 2-d array, y-coordinates of the data, same, shape as data
    initial_polynomial_coefficients: 1d array of the initial polynomial coefficients for the wavelength solution
    initial_tilt: float: initial angle measured clockwise of up in degrees
    initial_line_width: float: initial estimate of fwhm of the lines in angstroms
    lines: astropy table: must have the columns of catalog center in angstroms, and strength
    Returns
    -------
    best_fit_params: 1-d array: (best_fit_tilt, best_fit_line_width, *best_fit_polynomial_coefficients)
    """
    best_fit_params = maximize_match_filter((initial_tilt, initial_line_width, *initial_polynomial_coefficients), data,
                                            error, full_wavelength_solution_weights, (x, y), args=(lines,))
    return best_fit_params


class CalibrateWavelengths(Stage):
    """
    Stage that uses Arcs to fit wavelength solution
    """
    def do_stage(self, image):
        # for order in orders:
            # if no previous wavelength solution calculate it
                # copy order centers and get mask for height of a few extract median along axis=0
                # from 1D estimate linear solution
                # Estimate 1D distortion with higher order polynomials
            # Otherwise load wavelength solution
            # Fit 2D wavelength solution using initial guess either loaded or from 1D extraction
            # store fit info in table
            # evaluate wavelength solution at all pixels in 2D order
