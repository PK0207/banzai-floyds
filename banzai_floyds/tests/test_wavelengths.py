from banzai_floyds.wavelengths import gauss, linear_wavelength_solution, identify_peaks, correlate_peaks,\
    refine_peak_centers, full_wavelength_solution
import numpy as np
from astropy.table import Table
from numpy.polynomial.legendre import Legendre
from banzai_floyds.orders import order_region


def build_random_spectrum(seed=None, min_wavelength=3200, line_sigma=3, dispersion=2.5, nlines=10, nx=1001):
    # If given seed, use well behaved seed
    if seed:
        np.random.seed(seed)
    lines = Table({'wavelength': np.random.uniform(low=3500.0, high=5500.0, size=nlines),
                   'strength': np.random.uniform(low=0.0, high=1.0, size=nlines),
                   'line_source': ['Hg', 'Zn'] * (nlines // 2),
                   'used': [True] * nlines
                   },)

    input_spectrum = np.zeros(nx)

    # Why the coefficients in poly1d are in reverse order from numpy.polynomial.legendre is just beyond me
    input_wavelength_solution = np.poly1d((dispersion, min_wavelength))
    x_pixels = np.arange(nx)
    flux_scale = 1200

    # simulate a spectrum
    test_lines = []
    for line in lines:
        # And why roots is a property on poly1d objects and a method on numpy.polynomial.legendre. 🤦
        peak_center = (input_wavelength_solution - line['wavelength']).roots
        input_spectrum += line['strength'] * gauss(x_pixels, peak_center, line_sigma) * flux_scale
        test_lines.append(peak_center[0])
    return input_spectrum, lines, test_lines


def test_linear_wavelength_solution():

    min_wavelength = 3200
    dispersion = 2.5
    line_width = 3
    input_spectrum, lines, test_lines = build_random_spectrum(min_wavelength=min_wavelength, dispersion=dispersion,
                                                              line_sigma=line_width)

    linear_model = linear_wavelength_solution(input_spectrum, 0.01 * np.ones_like(input_spectrum), lines,
                                              dispersion, line_width, np.arange(4000, 5001))
    assert linear_model(0) == min_wavelength


def test_identify_peaks():
    # use well-behaved seed
    seed = 76856
    line_width = 3
    line_sep = 10
    input_spectrum, lines, test_lines = build_random_spectrum(seed=seed, line_sigma=line_width, nlines=6)

    recovered_peaks = identify_peaks(input_spectrum, 0.01 * np.ones_like(input_spectrum), line_width, line_sep)

    # Need to figure out how to handle blurred lines and combined peaks
    for peak in recovered_peaks:
        assert (peak in np.around(test_lines))


def test_correlate_peaks():
    min_wavelength = 3200
    dispersion = 2.5
    line_width = 3
    used_lines = 6
    input_spectrum, lines, test_peaks = build_random_spectrum(min_wavelength=min_wavelength, dispersion=dispersion,
                                                              line_sigma=line_width)

    linear_model = linear_wavelength_solution(input_spectrum, 0.01 * np.ones_like(input_spectrum), lines,
                                              dispersion, line_width, np.arange(4000, 5001))

    # find corresponding lines with lines missing
    match_threshold = 1
    corresponding_lines = correlate_peaks(np.array(test_peaks[:used_lines]), linear_model, lines, match_threshold)
    for corresponding_line in corresponding_lines:
        assert corresponding_line in lines["wavelength"][:used_lines]

    valid_line_count = len([cline for cline in corresponding_lines if cline])
    assert valid_line_count == used_lines

    # find corresponding lines with extra lines
    test_peaks_with_extra = np.concatenate((np.array(test_peaks[:used_lines]), np.random.uniform(0, 1000, 3)))
    match_threshold = 10
    corresponding_lines = correlate_peaks(test_peaks_with_extra, linear_model, lines, match_threshold)
    for corresponding_line in corresponding_lines:
        if corresponding_line:
            assert corresponding_line in lines["wavelength"][:used_lines]

    valid_line_count = len([cline for cline in corresponding_lines if cline])
    assert valid_line_count == used_lines


def test_refine_peak_centers():
    # use well-behaved seed
    seed = 75827
    line_width = 3
    line_sep = 10
    input_spectrum, lines, test_lines = build_random_spectrum(seed=seed, line_sigma=line_width)

    recovered_peaks = identify_peaks(input_spectrum, 0.01 * np.ones_like(input_spectrum), line_width, line_sep)

    fit_list = refine_peak_centers(input_spectrum, 0.01 * np.ones_like(input_spectrum), recovered_peaks, line_width)

    # Need to figure out how to handle blurred lines and overlapping peaks.
    for fit in fit_list:
        assert np.min(abs(test_lines - fit)) < 1


def test_2d_wavelength_solution():
    nx = 501
    data = np.zeros((512, nx))
    error = np.ones((512, nx))
    order_center = 151
    input_center_params = [order_center, 10, 20]
    order_height = 85
    trace_center = Legendre(input_center_params, domain=(0, data.shape[1] - 1))
    input_order_region = order_region(order_height, trace_center, data.shape)

    min_wavelength = 3200.0
    seed = 76856
    line_width = 3 * (2 * np.sqrt(2 * np.log(2)))
    dispersion = 2.5
    tilt = 15  # degrees
    input_spectrum, lines, test_lines = build_random_spectrum(seed=seed, line_sigma=3,
                                                              dispersion=dispersion, nlines=6, nx=nx)
    x1d = np.arange(data.shape[1], dtype=float)
    x2d, y2d = np.meshgrid(x1d, np.arange(data.shape[0], dtype=float))
    tilted_x = x2d + (y2d - trace_center(x1d)) * np.tan(np.deg2rad(tilt))
    data[input_order_region] = np.interp(tilted_x[input_order_region], x1d, input_spectrum)
    error[data >= 1.0] = 0.01 * data[data >= 1.0]

    # Convert between poly1d and legendre conventions
    converted_input_polynomial = Legendre((min_wavelength, dispersion), domain=(0, data.shape[1] - 1),
                                          window=(0, data.shape[1] - 1)).convert(domain=(0, data.shape[1] - 1))
    # Note that weight function has the line width in angstroms whereas our line width here is in pixels
    params = full_wavelength_solution(data[input_order_region], error[input_order_region], x2d[input_order_region],
                                      (y2d - trace_center(x1d))[input_order_region], converted_input_polynomial.coef,
                                      tilt, dispersion * line_width, lines)

    fit_tilt, fit_line_width, *fit_polynomial_coefficients = params
    # Assert that the best fit parameters are close to the inputs
    np.testing.assert_allclose(tilt, fit_tilt, atol=0.1)
    np.testing.assert_allclose(dispersion * line_width, fit_line_width, atol=0.3)
    np.testing.assert_allclose(converted_input_polynomial.coef, fit_polynomial_coefficients, atol=0.1)
