from matplotlib import pyplot as mp
from banzai_floyds.wavelengths import gauss, linear_wavelength_solution, identify_peaks, correlate_peaks,\
    refine_peak_centers
import numpy as np
from astropy.table import Table


def gaussian(x, mu, sig, strength):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) * strength


def build_random_spectrum(seed=None, min_wavelength=3200, line_width=3, dispersion=2.5):
    # If given seed, use well behaved seed
    if seed:
        np.random.seed(seed)
    lines = Table({'wavelength': np.random.uniform(low=3500.0, high=5500.0, size=10),
                   'strength': np.random.uniform(low=0.0, high=1.0, size=10),
                   'line_source': ['Hg', 'Zn'] * 5
                   },)

    nx = 1001
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
        input_spectrum += line['strength'] * gauss(x_pixels, peak_center, line_width) * flux_scale
        test_lines.append(peak_center[0])
    return input_spectrum, lines, test_lines


def test_1d_metric():
    # make a random list of lines
    line_list_length = 10
    rng = np.random.default_rng()
    line_strs = rng.random(line_list_length)
    peak_cntrs = np.sort(rng.random(line_list_length))
    peak_angstroms = peak_cntrs * 7000 + 3000
    test_lines = [{"wavelength": peak_cntr, 'line_strength': line_strs[i]}
                  for i, peak_cntr in enumerate(peak_angstroms)]

    # simulate a spectrum
    nx = 512
    data_1d = np.zeros(nx)

    # Add known lines
    peak_pixels = peak_cntrs * nx
    fwhm = 1
    flux_scale = 1200
    for peak, strength in zip(peak_pixels, line_strs):
        data_1d += gaussian(np.arange(nx), peak, fwhm, strength * flux_scale)
    # mp.plot(data_1d)

    # add extra lines
    extre_lines_num = 3
    unused_peaks = np.sort(rng.random(extre_lines_num)) * nx
    unused_str = rng.random(extre_lines_num)
    for peak, strength in zip(unused_peaks, unused_str):
        data_1d += gaussian(np.arange(nx), peak, fwhm, strength * flux_scale)

    # add continuum:
    data_1d += gaussian(np.arange(nx), nx // 2, nx // 10, flux_scale / 10)

    # Set the dispersion and some minor distortion

    # add noise to the spectrum
    noise_level = 30
    bias = 5
    data_1d += np.random.normal(bias, scale=noise_level, size=data_1d.shape)

    # Cross correlate the spectrum
    # Find the linear part of the wavelength solution

    # mp.plot(data_1d)
    # mp.show()
    pass


def test_linear_wavelength_solution():

    min_wavelength = 3200
    dispersion = 2.5
    line_width = 3
    input_spectrum, lines, test_lines = build_random_spectrum(min_wavelength=min_wavelength, dispersion=dispersion,
                                                              line_width=line_width)

    linear_model = linear_wavelength_solution(input_spectrum, 0.01 * np.ones_like(input_spectrum), lines,
                                              dispersion, line_width, np.arange(4000, 5001))
    assert linear_model(0) == min_wavelength


def test_identify_peaks():
    # use well-behaved seed
    seed = 76856
    line_width = 3
    line_sep = 10
    input_spectrum, lines, test_lines = build_random_spectrum(seed=seed, line_width=line_width)

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
                                                              line_width=line_width)

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
    input_spectrum, lines, test_lines = build_random_spectrum(seed=seed, line_width=line_width)

    recovered_peaks = identify_peaks(input_spectrum, 0.01 * np.ones_like(input_spectrum), line_width, line_sep)

    fit_list = refine_peak_centers(input_spectrum, 0.01 * np.ones_like(input_spectrum), recovered_peaks, line_width)

    # x_pixels = np.linspace(0, input_spectrum.size, 10000)
    # fit_spectrum = np.zeros(x_pixels.size)
    # for fit in fit_list:
    #     fit_spectrum += gauss(x_pixels, fit[0], fit[1], fit[2])
    # for peak in recovered_peaks:
    #     mp.plot([peak, peak], [-10, 200], color='red')
    #     mp.plot(input_spectrum)
    #     mp.plot(x_pixels, fit_spectrum, color="green")
    #     mp.xlim([peak-15, peak+15])
    #     mp.show()

    # Need to figure out how to handle blurred lines and overlapping peaks.
    for fit in fit_list:
        assert np.min(abs(test_lines - fit[0])) < 1
